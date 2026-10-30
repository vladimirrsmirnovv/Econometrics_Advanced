"""Нормализация названий регионов и населенных пунктов для сопоставления источников."""

import re
import unicodedata

LATIN_LOOKALIKES = str.maketrans("ABCEHKMOPTXaceopxy", "АВСЕНКМОРТХасеорху")

REGION_WORD_EXPANSIONS = {
    "обл": "область",
    "респ": "республика",
    "авт": "автономный",
    "автономная": "автономный",
    "ао": "автономный округ",
    "г": "город",
}

# Второе название субъекта, которое есть не во всех источниках.
REGION_OPTIONAL_WORDS = {"кузбасс", "чувашия"}

SETTLEMENT_TYPE_CLASSES = {
    "г": "город",
    "город": "город",
    "пгт": "поселок городского типа",
    "рп": "поселок городского типа",
    "гп": "поселок городского типа",
    "дп": "поселок городского типа",
    "кп": "поселок городского типа",
    "п": "поселок",
    "пос": "поселок",
    "поселок": "поселок",
    "с": "село",
    "д": "деревня",
    "дер": "деревня",
    "ст-ца": "станица",
    "х": "хутор",
    "аул": "аул",
    "сл": "слобода",
    "ст": "станция",
    "ж/д_ст": "станция",
    "п/ст": "станция",
    "м": "местечко",
    "нп": "населенный пункт",
    "сп": "сельский поселок",
    "г-к": "городок",
    "р-н": "район",
}

NOT_A_SETTLEMENT = {"район"}


def clean_text(text: str) -> str:
    """Приводит строку к нижнему регистру, заменяет латиницу-двойники и е с точками."""
    text = unicodedata.normalize("NFKC", str(text)).translate(LATIN_LOOKALIKES)
    return text.lower().replace("ё", "е").replace("\n", " ").strip()


def region_key(region_name: str) -> str:
    """Возвращает ключ региона: отсортированные слова названия без уточнений.

    Строки вида «область без автономных округов» получают метку `без_ао`,
    потому что это отдельная территория со своими показателями.
    """
    text = clean_text(region_name).replace("в том числе", " ")
    is_without_autonomous_okrugs = bool(re.search(r"\bкроме\b|\bбез\b", text))
    text = re.split(r"\(|\bбез\b|\bвключая\b|,", text)[0]
    words: set[str] = set()
    for word in re.findall(r"[а-яa-z0-9]+", text):
        words.update(REGION_WORD_EXPANSIONS.get(word, word).split())
    words -= REGION_OPTIONAL_WORDS
    key = " ".join(sorted(words))
    return f"{key} без_ао" if is_without_autonomous_okrugs else key


def is_federal_district(region_name: str) -> bool:
    """Проверяет, что строка относится к федеральному округу или стране целиком."""
    text = clean_text(region_name)
    return "федеральный округ" in text or "российская федерация" in text


def split_settlement(raw_name: str) -> tuple[str, str]:
    """Разделяет запись населенного пункта на собственно название и его тип.

    Тип записывается либо в конце строки («Ярославль г»), либо в начале
    («рп Рамонь»), изредка с обеих сторон. Название возвращается в нижнем
    регистре. Если тип не распознан, выбрасывается ошибка, чтобы неизвестное
    обозначение не прошло в данные молча.
    """
    words = clean_text(raw_name).replace("населенный пункт", "нп").split()
    types = [word.rstrip(".") for word in words]
    if len(words) > 1 and types[-1] in SETTLEMENT_TYPE_CLASSES:
        name_words, settlement_type = words[:-1], types[-1]
    elif len(words) > 1 and types[0] in SETTLEMENT_TYPE_CLASSES:
        name_words, settlement_type = words[1:], types[0]
    else:
        raise ValueError(f"Не распознан тип населенного пункта: {raw_name!r}")
    if len(name_words) > 1 and name_words[0].rstrip(".") in SETTLEMENT_TYPE_CLASSES:
        name_words = name_words[1:]
    return " ".join(name_words), settlement_type


def settlement_base_name(name: str) -> str:
    """Сводит название к сравнимому виду: без скобок, пробелов, дефисов и е с точками."""
    text = re.sub(r"\([^)]*\)", " ", clean_text(name))
    return re.sub(r"[^а-яa-z0-9]+", "", text)


def settlement_type_class(settlement_type: str) -> str:
    """Возвращает класс типа населенного пункта, объединяющий синонимы."""
    return SETTLEMENT_TYPE_CLASSES[settlement_type.rstrip(".").lower()]
