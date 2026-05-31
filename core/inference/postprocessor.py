# xu li sau khi model predict xong, bao gồm:
# - Bầu chọn nhãn (Majority Voting)
# - Ngưỡng tin cậy (Confidence Thresholding để chống False Positives)
# - Đè nhãn (Rule-based Loitering)
# core/inference/postprocessor.py


from collections import Counter
from collections import deque
from typing import Tuple

class BehaviorPostprocessor:
    def __init__(self, loitering_threshold_sec: float = 10.0):
        self.loitering_threshold_sec = loitering_threshold_sec

    def process(self, raw_label: str, raw_conf: float, recent_preds: deque, time_tracked: float) -> Tuple[str, float]:
        """Xử lý mượt nhãn, áp dụng các ngưỡng an toàn và luật Loitering"""
        
        # 1. Bầu chọn nhãn (Majority Voting)
        recent_preds.append((raw_label, raw_conf))
        labels_in_queue = [pred[0] for pred in recent_preds]
        most_common_label = Counter(labels_in_queue).most_common(1)[0][0]
        
        confs_of_winner = [pred[1] for pred in recent_preds if pred[0] == most_common_label]
        final_conf = sum(confs_of_winner) / len(confs_of_winner)
        final_label = most_common_label

        # 2. Ngưỡng tin cậy (Confidence Thresholding để chống False Positives)
        if final_label == "fighting" and final_conf < 0.6:
            final_label = "normal"
        elif final_label == "falling" and final_conf < 0.5:
            final_label = "normal"

        # 3. Đè nhãn (Rule-based Loitering)
        if time_tracked > self.loitering_threshold_sec and final_label == "normal":
            final_label = "loitering"
            final_conf = 1.0 # Tự tin 100%

        return final_label, final_conf