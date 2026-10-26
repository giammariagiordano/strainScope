"""Dictionary markers: belonging (we vs I) and gratitude."""

from app.psych import grateful, pronouns, reactions


def test_pronouns_count_we_and_i_forms():
    assert pronouns("We should fix our tests; I think my patch helps. Let's merge.") == (12, 3, 2)
    assert pronouns("I’m not sure") == (3, 0, 1)  # typographic apostrophe
    assert pronouns("") == (0, 0, 0)


def test_gratitude_phrases_and_emoji():
    assert grateful("Thanks a lot!") and grateful("thank you for this") and grateful("Nice catch") and grateful("🙏")
    assert grateful("Much appreciated")
    assert not grateful("This is a nice feature") and not grateful("Work in progress")


def test_reactions_split_positive_and_negative():
    assert reactions({"+1": 3, "heart": 1, "-1": 2, "laugh": 5, "eyes": 1, "total_count": 12}) == (4, 2)
    assert reactions(None) == (None, None)
