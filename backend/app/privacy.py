import hashlib
import hmac
import secrets

from sqlalchemy.orm import Session

from .config import settings
from .models import AppState


def get_salt(db: Session) -> str:
    if settings.pseudonym_salt:
        return settings.pseudonym_salt
    state = db.get(AppState, "pseudonym_salt")
    if state is None:
        state = AppState(key="pseudonym_salt", value={"salt": secrets.token_hex(16)})
        db.add(state)
        db.flush()
    return state.value["salt"]


ADJECTIVES = (
    "amber azure brisk calm coral crisp dusky eager fable fern frost gentle golden hazel ivory jade "
    "keen lunar mellow misty noble ochre pale quiet rapid rosy russet sage silver solar teal umber "
    "vivid warm witty zesty"
).split()
NOUNS = (
    "badger beacon birch cedar comet crane delta ember falcon finch fjord gecko harbor heron iris "
    "kestrel lark lynx maple meadow orca otter pebble quartz raven reef robin sparrow summit tern "
    "tide willow wren yarrow"
).split()


def pseudonym(salt: str, canonical_key: str) -> str:
    """Stable, human-friendly pseudonym such as `amber-heron-3f9a`."""
    digest = hmac.new(salt.encode(), canonical_key.encode(), hashlib.sha256).digest()
    adj = ADJECTIVES[digest[0] % len(ADJECTIVES)]
    noun = NOUNS[digest[1] % len(NOUNS)]
    return f"{adj}-{noun}-{digest[2:4].hex()}"
