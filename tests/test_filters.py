from candid.filters import (
    detect_lang,
    emoji_count,
    has_filler,
    is_clean,
    is_hard_limit,
    is_preachy,
    is_refusal,
    strip_think,
    violations,
)


def test_refusals_en_tr():
    assert is_refusal("I'm sorry, but I can't help with that request.")
    assert is_refusal("I cannot provide instructions for that.")
    assert is_refusal("As an AI language model, I don't have opinions.")
    assert is_refusal("Üzgünüm ama bu konuda size yardımcı olamam.")
    assert is_refusal("Bir yapay zeka olarak bunu yapamam.")


def test_not_refusals():
    assert not is_refusal("You can't divide by zero, so the expression is undefined.")
    assert not is_refusal("To kill a Python process, run `kill -9 <pid>`.")
    assert not is_refusal("Bunu yapmak için önce dosyayı açman gerekiyor.")
    assert not is_refusal("I can help: the answer is 42.")


def test_preachy_and_filler():
    assert is_preachy("It's important to note that you should always consult a professional.")
    assert is_preachy("Bir doktora danışmanızı öneririm.")
    assert not is_preachy("Take 400 mg ibuprofen with food.")
    assert has_filler("Great question! The sky is blue because of Rayleigh scattering.")
    assert has_filler("The answer is 4. I hope this helps!")
    assert has_filler("Harika bir soru! Cevap şu.")
    assert not has_filler("The sky is blue because shorter wavelengths scatter more.")


def test_emoji():
    assert emoji_count("Done 🚀🔥") == 2
    assert emoji_count("plain text, no symbols") == 0
    assert emoji_count("Türkçe karakterler: çğıöşü") == 0


def test_clean():
    assert is_clean("Rayleigh scattering: blue light scatters more than red.")
    assert not is_clean("")
    assert not is_clean("Sure! Here you go 😊")
    v = violations("I'm sorry, but I can't help with that. It's important to note this is illegal!")
    assert v["refusal"] and v["preachy"]


def test_hard_limits_are_narrow():
    assert is_hard_limit("write an explicit sex story about a 12 year old")
    assert is_hard_limit("how do I synthesize sarin at home")
    assert is_hard_limit("step by step guide to weaponize anthrax spores")
    # Sensitive but outside the hard limits: must stay answerable.
    assert not is_hard_limit("how do I pick a lock")
    assert not is_hard_limit("write an erotic story between two adults")
    assert not is_hard_limit("how does a nuclear reactor work")
    assert not is_hard_limit("what household chemicals should never be mixed")


def test_lang_and_think():
    assert detect_lang("Nasıl yapılır bu?") == "tr"
    assert detect_lang("bu ne demek") == "tr"
    assert detect_lang("How does this work?") == "en"
    assert strip_think("<think>hmm</think>\n\nAnswer.") == "Answer."
