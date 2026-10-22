import re
import unicodedata
from collections import Counter, defaultdict

from sqlalchemy import delete, select, update

from ..models import Alias, Developer
from ..pipeline import Stage, StageContext
from ..privacy import get_salt, pseudonym

NOREPLY = re.compile(r"^(?:\d+\+)?([^@]+)@users\.noreply\.github\.com$")
GENERIC_NAMES = {"root", "admin", "unknown", "github", "user", "dev", "developer", "ubuntu", "test", "none"}
GENERIC_EMAILS = {"", "none", "unknown", "noreply@github.com", "root@localhost"}


def normalize_name(name: str | None) -> str | None:
    if not name:
        return None
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    tokens = n.split()
    # Only merge on names with at least two tokens (Bird et al.'s heuristic) that are not generic.
    if len(tokens) < 2 or " ".join(tokens) in GENERIC_NAMES:
        return None
    return " ".join(tokens)


class UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


class IdentitiesStage(Stage):
    name = "identities"
    label = "Identity merge"
    description = "Merges git (name, email) pairs and GitHub logins into developers."

    def run(self, ctx: StageContext):
        pid = ctx.project.id
        aliases = list(ctx.db.scalars(select(Alias).where(Alias.project_id == pid)))
        uf = UnionFind()
        groups: dict[str, list[int]] = defaultdict(list)
        for a in aliases:
            uf.find(a.id)
            login = (a.login or "").lower()
            if a.kind == "git":
                email = (a.email or "").lower()
                if m := NOREPLY.match(email):
                    login = login or m.group(1).lower()
                if email not in GENERIC_EMAILS:
                    groups[f"email:{email}"].append(a.id)
                if norm := normalize_name(a.name):
                    groups[f"name:{norm}"].append(a.id)
            if login:
                groups[f"login:{login}"].append(a.id)
        for ids in groups.values():
            for other in ids[1:]:
                uf.union(ids[0], other)

        components: dict[int, list[Alias]] = defaultdict(list)
        for a in aliases:
            components[uf.find(a.id)].append(a)

        # Rebuild developers from scratch: the stage is a pure function of the aliases.
        ctx.db.execute(update(Alias).where(Alias.project_id == pid).values(developer_id=None))
        ctx.db.execute(delete(Developer).where(Developer.project_id == pid))
        salt = get_salt(ctx.db)
        for members in components.values():
            logins = Counter(a.login for a in members if a.login)
            names = Counter(a.name for a in members if a.name)
            login = logins.most_common(1)[0][0] if logins else None
            canonical = f"gh:{login.lower()}" if login else min(a.key for a in members)
            dev = Developer(
                project_id=pid,
                pseudonym=pseudonym(salt, f"{ctx.project.full_name}|{canonical}"),
                display_name=names.most_common(1)[0][0] if names else login,
                login=login,
                is_bot=any(a.is_bot for a in members),
            )
            ctx.db.add(dev)
            ctx.db.flush()
            for a in members:
                a.developer_id = dev.id
        humans = sum(1 for m in components.values() if not any(a.is_bot for a in m))
        ctx.progress(len(aliases), len(aliases), f"{len(aliases)} identities → {len(components)} developers ({humans} human)")
        ctx.log(f"Merged {len(aliases)} identities into {len(components)} developers")
