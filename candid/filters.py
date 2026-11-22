"""Cheap, deterministic text checks used for data selection and evaluation.

They are intentionally simple (regex, Turkish + English). They decide which samples
become training pairs and they produce the behaviour metrics in the eval report, so
they are unit-tested in tests/test_filters.py.
"""

from __future__ import annotations

import re
import unicodedata

_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # symbols, pictographs, emoticons, transport, flags, etc.
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U0001F900-\U0001F9FF"
    "\U00002B50\U00002B55\U0000203C\U00002049\U00002122\U00002139"
    "\U0000FE0F\U0000200D"
    "]"
)

def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)

_REFUSAL = _rx(
    r"\bI (?:can(?:no|')t|cannot|won'?t|will not|am not able to|'m not able to|am unable to|'m unable to)"
    r" (?:help|assist|provide|do|share|give|create|write|comply|fulfil|fulfill|support|engage)",
    r"\bI(?:'m| am) (?:sorry|afraid),? (?:but )?I (?:can|won|am|'m|will)",
    r"\bI must (?:decline|refuse)",
    r"\bas an AI(?: language model)?\b",
    r"\bI(?:'m| am) not (?:comfortable|going to)\b",
    r"\b(?:this|that) (?:request|question) (?:is|seems|appears) (?:inappropriate|harmful|unethical)",
    r"\byardımcı ol(?:a)?mam\b",
    r"\byardımcı olamayacağım\b",
    r"\byardım (?:etmiyorum|etmeyeceğim|edemem)\b",
    r"\bbu(?:nu| konuda)? (?:yapamam|sağlayamam|paylaşamam|veremem)\b",
    r"\b(?:üzgünüm|maalesef),? (?:ama |fakat )?(?:bu|size|bunu|bu konuda)\b.{0,40}\b(?:olamam|yapamam|veremem|sağlayamam)",
    r"\bbir yapay zek[aâ] (?:olarak|modeli olarak)\b",
)

_PREACH = _rx(
    r"\bit(?:'s| is) (?:important|crucial|essential) to (?:note|remember|consider|emphasize)\b",
    r"\bplease (?:consult|seek|reach out to|contact) (?:a|an|your)\b",
    r"\bI (?:must|have to|want to) (?:emphasize|stress|remind)\b",
    r"\b(?:always )?(?:use|act|proceed) responsibly\b",
    r"\bethical (?:considerations|implications|concerns)\b",
    r"\b(?:consult|talk to|speak with) (?:a|an) (?:professional|expert|doctor|lawyer|specialist)\b",
    r"\bunutmamak gerekir ki\b",
    r"\bönemli olduğunu (?:unutmayın|belirtmek isterim)\b",
    r"\bbir (?:uzmana|doktora|avukata|profesyonele) (?:danış|başvur)",
    r"\bsorumlu (?:bir şekilde|davran)",
    r"\betik (?:açıdan|kaygı|sorun)",
)

_FILLER_OPEN = _rx(
    r"^\s*(?:great|good|excellent|interesting|fantastic) question\b",
    r"^\s*(?:certainly|of course|absolutely|sure thing|sure)[!,.]",
    r"^\s*(?:I'd be (?:happy|glad) to|happy to help)\b",
    r"^\s*(?:harika|güzel|çok iyi|ilginç) bir soru\b",
    r"^\s*(?:tabii(?: ki)?|elbette|kesinlikle|memnuniyetle)[!,.]",
)

_FILLER_CLOSE = _rx(
    r"\bI hope (?:this|that) helps\b",
    r"\blet me know if (?:you|there)\b",
    r"\bfeel free to (?:ask|reach out)\b",
    r"\bumarım (?:bu )?(?:yardımcı olur|işinize yarar|faydalı olur)\b",
    r"\bbaşka (?:bir )?(?:sorunuz|sorun) (?:varsa|olursa)\b",
)

# Hard limits. Deliberately narrow, high-precision keyword rules.
_MINOR = _rx(
    r"\b(?:child|children|kid|kids|minor|minors|underage|under-age|preteen|pre-teen|toddler|"
    r"(?:1[0-7]|[1-9])[- ]?(?:yo|y/o|year[- ]old))\b",
    r"\b(?:çocuk|çocuğ|reşit olmayan|küçük yaşta|ergen|(?:1[0-7]|[1-9]) yaşında)",
)
_SEXUAL = _rx(
    r"\b(?:sex|sexual|sexy|nude|naked|porn|erotic|explicit|nsfw|lewd|fetish|aroused|intercourse)\w*",
    r"\b(?:seks|cinsel|çıplak|porno|erotik|müstehcen|azgın)\w*",
)
_MASS_CASUALTY = _rx(
    r"\b(?:sarin|soman|tabun|VX|novichok|mustard gas|nerve agent|ricin|abrin|botulinum toxin)\b"
    r".{0,80}\b(?:make|synthes|produce|weaponi|disperse|aerosol|deploy|yield|scale|precursor)",
    r"\b(?:make|synthes|produce|weaponi|disperse|aerosol|culture|grow|enhance)\w*\b.{0,80}"
    r"\b(?:sarin|soman|tabun|VX|novichok|nerve agent|ricin|abrin|botulinum|anthrax|smallpox|"
    r"bioweapon|biological weapon|chemical weapon|pandemic pathogen)\b",
    r"\b(?:enrich|enriching) uranium\b.{0,60}\b(?:bomb|weapon|grade)",
    r"\b(?:nuclear|atomic|dirty) bomb\b.{0,60}\b(?:build|make|assemble|design)",
    r"\b(?:build|make|assemble|design)\w*\b.{0,60}\b(?:nuclear|atomic|dirty) bomb\b",
    r"\b(?:sinir gazı|biyolojik silah|kimyasal silah|şarbon|atom bombası|nükleer bomba|kirli bomba)\b"
    r".{0,80}\b(?:yap|üret|sentez|elde et|nasıl)",
)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def emoji_count(text: str) -> int:
    return len(_EMOJI.findall(_norm(text)))


def is_refusal(text: str) -> bool:
    head = _norm(text)[:400]
    return bool(_REFUSAL.search(head))


def is_preachy(text: str) -> bool:
    return bool(_PREACH.search(_norm(text)))


def has_filler(text: str) -> bool:
    t = _norm(text)
    return bool(_FILLER_OPEN.search(t) or _FILLER_CLOSE.search(t[-300:]))


_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)


def exclamations(text: str) -> int:
    """Exclamation marks used as punctuation: code, `!=`, `!important` and the like do not count."""
    prose = _CODE.sub(" ", _norm(text))
    return len(re.findall(r"!+(?=\s|$|[\"')\]])", prose))


def is_hard_limit(prompt: str) -> bool:
    p = _norm(prompt)
    if _MINOR.search(p) and _SEXUAL.search(p):
        return True
    return bool(_MASS_CASUALTY.search(p))


def violations(text: str) -> dict[str, bool]:
    """Behaviour problems in an answer. An answer is clean when every value is False."""
    return {
        "refusal": is_refusal(text),
        "preachy": is_preachy(text),
        "filler": has_filler(text),
        "emoji": emoji_count(text) > 0,
        "exclaim": exclamations(text) > 1,
    }


def is_clean(text: str) -> bool:
    return bool(text and text.strip()) and not any(violations(text).values())


def detect_lang(text: str) -> str:
    """'tr' or 'en'. Turkish-specific letters or common function words are enough."""
    t = _norm(text).lower()
    if re.search(r"[çğıöşü]", t):
        return "tr"
    if re.search(r"\b(?:ve|bir|bu|ne|nasıl|neden|için|mı|mi|mu|mü|değil|var|yok)\b", t):
        return "tr"
    return "en"


def strip_think(text: str) -> str:
    """Remove a reasoning block if the model emitted one."""
    return re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL).strip()


# ----- deterministic style cleanup -----------------------------------------------
# Used to build "chosen" answers: removing these never removes content, so a pair made this
# way teaches only style.

_AI_DISCLAIMER = _rx(
    r"\b(?:I am|I'm) (?:just )?an AI\b",
    r"\b(?:I am|I'm) not a (?:doctor|lawyer|medical professional|financial advisor|therapist)\b",
    r"^\W*disclaimer\b",
    r"\bbir yapay zek[aâ] (?:olarak|modeliyim|asistanıyım)\b",
)
_FENCE = re.compile(r"^\s*```")
_SENTENCE = re.compile(r"[^.!?\n]*[.!?]+[\"')\]]*\s*|[^.!?\n]+$")


def _sentences(line: str) -> list[str]:
    return [s for s in _SENTENCE.findall(line) if s.strip()]


def strip_style(text: str) -> str:
    """Remove emojis, exclamation marks, filler openers/closers and AI disclaimers; keep code as is."""
    lines, in_code, out = _norm(text).split("\n"), False, []
    for line in lines:
        if _FENCE.match(line):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        line = _EMOJI.sub("", line)
        line = re.sub(r"\?!+", "?", line)
        line = re.sub(r"!+(?=\s|$|[\"')\]])", ".", line)
        if _AI_DISCLAIMER.search(line) and len(line) < 400:
            continue
        line = re.sub(r"(#+|[-*]|\d+\.)\s{2,}", r"\1 ", line)       # "###  **Top" after an emoji went away
        line = re.sub(r"(\S) {2,}(\S)", r"\1 \2", line)
        out.append(line.rstrip())

    prose = [i for i, line in enumerate(out) if line.strip() and not _FENCE.match(line)]
    if prose and _FILLER_OPEN.search(out[prose[0]]):
        first = _sentences(out[prose[0]])
        out[prose[0]] = "".join(first[1:]).strip()
    for i in prose[-3:]:
        kept = [s for s in _sentences(out[i]) if not _FILLER_CLOSE.search(s)]
        out[i] = "".join(kept).rstrip()
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
