import csv
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path


LABEL_AUTHENTIC = 0
LABEL_COMPUTER = 1
MODEL_VERSION = "1.0.0"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return re.findall(r"[a-z0-9']+", clean_text(text))


def token_features(text):
    words = tokenize(text)
    features = list(words)
    features.extend(f"{a} {b}" for a, b in zip(words, words[1:]))
    return features


def split_sentences(text):
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return sentences or [text.strip()] if text.strip() else [""]


def extract_indicators(text):
    words = tokenize(text)
    sentences = split_sentences(text)
    num_words = max(len(words), 1)
    sent_lengths = [len(tokenize(sentence)) for sentence in sentences] or [0]
    avg_sent_len = sum(sent_lengths) / len(sent_lengths)
    variance = sum((length - avg_sent_len) ** 2 for length in sent_lengths) / len(sent_lengths)
    unique_words = len(set(words))
    repeated_words = sum(count for count in Counter(words).values() if count > 1)
    first_person = {"i", "me", "my", "mine", "myself"}
    hype_words = {
        "amazing", "perfect", "best", "excellent", "awesome", "incredible",
        "wonderful", "fantastic", "flawless", "life-changing", "highly",
    }

    positive = {
        "good", "great", "excellent", "amazing", "perfect", "love", "loved",
        "best", "wonderful", "fantastic", "recommend", "awesome", "helpful",
    }
    negative = {
        "bad", "poor", "terrible", "awful", "worst", "broken", "disappointed",
        "return", "refund", "late", "cheap", "problem", "issue", "fake",
    }
    sentiment_hits = sum(1 for word in words if word in positive) - sum(
        1 for word in words if word in negative
    )

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_sent_len, 3),
        "sentence_length_variation": round(math.sqrt(variance), 3),
        "type_token_ratio": round(unique_words / num_words, 3),
        "exclamation_ratio": round(text.count("!") / max(len(sentences), 1), 3),
        "question_count": text.count("?"),
        "first_person_ratio": round(
            sum(1 for word in words if word in first_person) / num_words, 3
        ),
        "repetition_ratio": round(repeated_words / num_words, 3),
        "capital_ratio": round(sum(1 for char in text if char.isupper()) / max(len(text), 1), 3),
        "hype_word_ratio": round(sum(1 for word in words if word in hype_words) / num_words, 3),
        "sentiment_intensity": round(abs(sentiment_hits) / num_words, 3),
    }


class DeceptionModel:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_doc_counts = Counter()
        self.class_feature_counts = defaultdict(Counter)
        self.class_total_features = Counter()
        self.vocabulary = set()
        self.trained_rows = 0
        self.source = "unknown"

    def fit(self, rows, source="training data"):
        for text, label in rows:
            label = int(label)
            self.class_doc_counts[label] += 1
            self.trained_rows += 1
            for feature in token_features(text):
                self.class_feature_counts[label][feature] += 1
                self.class_total_features[label] += 1
                self.vocabulary.add(feature)
        self.source = source
        if not self.class_doc_counts:
            raise ValueError("No training rows were provided.")
        return self

    def _log_probability(self, text, label):
        total_docs = sum(self.class_doc_counts.values())
        class_docs = self.class_doc_counts[label]
        log_prob = math.log((class_docs + self.alpha) / (total_docs + 2 * self.alpha))
        denominator = self.class_total_features[label] + self.alpha * max(len(self.vocabulary), 1)
        for feature in token_features(text):
            count = self.class_feature_counts[label][feature]
            log_prob += math.log((count + self.alpha) / denominator)
        return log_prob

    def predict_proba(self, text):
        if not self.class_doc_counts:
            raise ValueError("Model is not trained.")
        log_auth = self._log_probability(text, LABEL_AUTHENTIC)
        log_comp = self._log_probability(text, LABEL_COMPUTER)
        max_log = max(log_auth, log_comp)
        auth = math.exp(log_auth - max_log)
        comp = math.exp(log_comp - max_log)
        lexical_score = comp / (auth + comp)
        style_score = style_deception_score(text)
        return 0.78 * lexical_score + 0.22 * style_score

    def predict(self, text, threshold=0.5):
        score = self.predict_proba(text)
        return {
            "deception_score": round(score, 4),
            "prediction": "Likely computer-generated" if score >= threshold else "Likely authentic",
            "label": LABEL_COMPUTER if score >= threshold else LABEL_AUTHENTIC,
            "confidence": round(max(score, 1 - score), 4),
            "key_indicators": extract_indicators(text),
            "model_source": self.source,
            "trained_rows": self.trained_rows,
            "model_version": MODEL_VERSION,
        }


def style_deception_score(text):
    indicators = extract_indicators(text)
    score = 0.0
    if indicators["word_count"] >= 45:
        score += 0.14
    if indicators["type_token_ratio"] > 0.78:
        score += 0.16
    if indicators["sentence_length_variation"] < 3 and indicators["sentence_count"] >= 3:
        score += 0.14
    if indicators["exclamation_ratio"] >= 1:
        score += 0.16
    if indicators["first_person_ratio"] < 0.015 and indicators["word_count"] >= 25:
        score += 0.12
    if indicators["hype_word_ratio"] > 0.035:
        score += 0.16
    if indicators["sentiment_intensity"] > 0.06:
        score += 0.12
    return min(max(score, 0.05), 0.95)


def load_csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = {name.lower().strip(): name for name in reader.fieldnames or []}
        text_col = next(
            (fieldnames[name] for name in ("text", "text_", "review", "content", "message")
             if name in fieldnames),
            None,
        )
        label_col = next(
            (fieldnames[name] for name in ("label", "target", "class", "category")
             if name in fieldnames),
            None,
        )
        if not text_col or not label_col:
            raise ValueError("CSV must include text/text_/review and label/target columns.")

        rows = []
        for row in reader:
            text = (row.get(text_col) or "").strip()
            raw_label = (row.get(label_col) or "").strip().upper()
            if not text or not raw_label:
                continue
            if raw_label in {"CG", "COMPUTER", "COMPUTER-GENERATED", "AI", "1", "DECEPTIVE", "FAKE"}:
                label = LABEL_COMPUTER
            elif raw_label in {"OR", "ORIGINAL", "HUMAN", "0", "GENUINE", "AUTHENTIC", "REAL"}:
                label = LABEL_AUTHENTIC
            else:
                continue
            rows.append((text, label))
        return rows


def demo_rows():
    authentic = [
        "The delivery came a day late, but the product works fine. I used it twice and the build feels okay for the price.",
        "I bought this for my kitchen. The handle is comfortable, though the lid is a little loose.",
        "The hotel room was clean and the staff helped me change rooms when the AC made noise.",
        "I liked the shoes at first, but after two weeks the sole started wearing down near the heel.",
        "My order arrived with one missing cable. Customer support replied the next morning and sent a replacement.",
        "The app is useful, although the setup took me a few tries because the instructions skipped one step.",
    ]
    generated = [
        "This product is absolutely amazing! It exceeded all my expectations and I highly recommend it to everyone!",
        "An excellent experience from start to finish. The quality is perfect, the design is flawless, and the value is unbeatable.",
        "I am extremely satisfied with this wonderful item. It is the best purchase I have ever made and deserves five stars.",
        "Fantastic quality, incredible service, and perfect performance. Anyone looking for a premium option should buy this immediately!",
        "This hotel was wonderful in every possible way. The rooms were immaculate, the service was exceptional, and everything was perfect.",
        "A truly outstanding product with amazing features, excellent durability, and flawless results. Highly recommended!",
    ]
    return [(text, LABEL_AUTHENTIC) for text in authentic] + [
        (text, LABEL_COMPUTER) for text in generated
    ]


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(model, handle)


def load_model(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)
