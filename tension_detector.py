from dataclasses import dataclass


@dataclass
class TensionResult:
    score: float
    tense: bool


class TensionDetector:
    """
    Yüz ölçümlerinden basit gerginlik skoru üretir.

    İlk prototipte skor dışarıdan verilen ölçümlerle hesaplanır.
    Sonraki adımda MediaPipe yüz noktalarına bağlanacaktır.
    """

    def __init__(
        self,
        threshold: float = 0.62,
        required_frames: int = 12
    ) -> None:
        self.threshold = threshold
        self.required_frames = required_frames
        self.high_score_frames = 0

    def update(
        self,
        brow_distance_ratio: float,
        brow_lowering_ratio: float,
        lip_compression_ratio: float
    ) -> TensionResult:

        brow_close_score = self._inverse_normalize(
            brow_distance_ratio,
            low=0.10,
            high=0.22
        )

        brow_lower_score = self._normalize(
            brow_lowering_ratio,
            low=0.08,
            high=0.22
        )

        lip_score = self._inverse_normalize(
            lip_compression_ratio,
            low=0.015,
            high=0.060
        )

        score = (
            0.40 * brow_close_score +
            0.30 * brow_lower_score +
            0.30 * lip_score
        )

        if score >= self.threshold:
            self.high_score_frames += 1
        else:
            self.high_score_frames = max(0, self.high_score_frames - 2)

        tense = self.high_score_frames >= self.required_frames

        return TensionResult(
            score=round(score, 3),
            tense=tense
        )

    @staticmethod
    def _normalize(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0

        result = (value - low) / (high - low)
        return max(0.0, min(1.0, result))

    @staticmethod
    def _inverse_normalize(
        value: float,
        low: float,
        high: float
    ) -> float:
        return 1.0 - TensionDetector._normalize(value, low, high)

    def reset(self) -> None:
        self.high_score_frames = 0
