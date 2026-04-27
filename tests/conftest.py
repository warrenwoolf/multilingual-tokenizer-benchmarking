"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MULTILINGUAL_SAMPLES = [
    # English
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "She sells seashells by the seashore every summer afternoon before sunset.",
    "Machine learning models tokenize text before training language representations.",
    "Natural language processing requires careful attention to morphology and syntax.",
    "Tokenizers split input text into subword units that the model can process.",
    "Researchers compare algorithms across multiple languages and vocabulary sizes.",
    "Effective benchmarks reveal systematic differences between tokenizer families.",
    "A well-designed experiment controls for corpus size, vocabulary, and evaluation metric.",
    # Russian
    "Быстрая коричневая лиса прыгает через ленивую собаку каждый вечер у реки.",
    "Москва — столица России, крупнейший город по численности населения страны.",
    "Обработка естественного языка требует хороших токенизаторов для морфологии.",
    "Исследователи сравнивают алгоритмы токенизации на разных языках мира.",
    "Подбор размера словаря влияет на качество языковой модели в целом.",
    "Русский язык обладает богатой морфологией и сложной системой склонений.",
    # Hindi
    "नमस्ते दुनिया, यह एक परीक्षण वाक्य है जो हिंदी में लिखा गया है।",
    "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली और लिखी जाती हैं।",
    "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है हर सुबह।",
    "प्राकृतिक भाषा प्रसंस्करण के लिए अच्छे टोकनाइज़र की आवश्यकता होती है।",
    "शोधकर्ता विभिन्न भाषाओं में टोकनाइज़ेशन एल्गोरिदम की तुलना करते हैं।",
    "हिंदी एक समृद्ध आकारिकी वाली भाषा है जिसमें अनेक रूप मौजूद हैं।",
    # Turkish
    "Hızlı kahverengi tilki tembel köpeğin üzerinden her sabah atlar ve koşar.",
    "Türkiye, Avrupa ve Asya kıtalarını birbirine bağlayan önemli bir ülkedir.",
    "Doğal dil işleme, morfolojik olarak zengin diller için oldukça zordur.",
    "Araştırmacılar farklı diller üzerinde tokenlaştırma algoritmalarını karşılaştırır.",
    "Sözlük boyutunun seçimi, dil modelinin kalitesini doğrudan etkileyen bir faktördür.",
    "Türkçe sondan eklemeli bir dil olduğu için kelime yapıları oldukça karmaşıktır.",
]


@pytest.fixture(scope="session")
def tiny_corpus(tmp_path_factory) -> Path:
    """Write a ~200KB multilingual corpus file and return its path.

    Built by repeating the MULTILINGUAL_SAMPLES list. Fully offline; no network.
    """
    path = tmp_path_factory.mktemp("corpus") / "tiny.txt"
    repeats = 200
    lines = MULTILINGUAL_SAMPLES * repeats
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def tmp_artifact_dir(tmp_path) -> Path:
    """Per-test artifact directory."""
    d = tmp_path / "artifact"
    d.mkdir()
    return d


SAMPLE_STRINGS = [
    "Hello, world!",
    "Привет, мир!",
    "नमस्ते दुनिया",
    "Merhaba dünya",
    "mixed: English + Русский + हिन्दी + Türkçe",
    "emoji test",
]
