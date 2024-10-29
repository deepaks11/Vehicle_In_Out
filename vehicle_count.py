import cv2
import numpy as np
import supervision as sv
from polygan_line import LineIntersectionTest
from supervision.geometry.core import Position


class VehicleCount:

    def __init__(self, model):
        self.model = model
        self.person_count_in = 0
        self.person_count_out = 0
        self.tracked_ids = {}  # To track positions of persons
        self.in_id = {}
        self.out_id = {}
        self.entry_zone = 5  # Pixels above the line for "In"
        self.exit_zone = -5  # Pixels below the line for "Out"

    def check_side(self, bbox_center, line_coord):
        """Check if the center of the bbox is above or below the line."""
        line_start, line_end = line_coord
        return (line_end[0] - line_start[0]) * (bbox_center[1] - line_start[1]) - (line_end[1] - line_start[1]) * (bbox_center[0] - line_start[0])

    def predict(self, q_img, line_coord):
        try:
            frame = q_img.get()

            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator(text_padding=8, text_scale=1, text_thickness=2)

            result = self.model.track(source=frame, imgsz=1280, conf=0.5, persist=True, verbose=False, classes=[1, 2, 3, 4, 5, 6, 7, 8], tracker="botsort.yaml")

            result = result[0]
            detections = sv.Detections.from_ultralytics(result)

            if detections:
                labels = [
                    f"{tracker_id} {self.model.names[class_id]}"
                    for box, mask, confidence, class_id, tracker_id, class_name
                    in detections
                ]
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
                entry_line_start = line_coord[0]  # Start point of the line
                entry_line_end = line_coord[1]    # End point of the line

                for i in range(len(detections.xyxy)):
                    # Get the bounding box coordinates for the current detection
                    bbox = detections.xyxy[i]  # bbox will be an array like [x_min, y_min, x_max, y_max]

                    # Calculate the center of the bounding box
                    bbox_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

                    if result.boxes.id is not None:
                        tracker_id = detections.tracker_id[i]  # Access the tracker ID for the current detection

                        # Initialize tracking for new persons
                        if tracker_id not in self.tracked_ids:
                            # New entry, track this person's state
                            self.tracked_ids[tracker_id] = {"state": "none"}
                            self.in_id[tracker_id] = {"count": 0}
                            self.out_id[tracker_id] = {"count": 0}
                            continue

                        # Get the current tracked info
                        tracked_info = self.tracked_ids[tracker_id]
                        state = tracked_info["state"]

                        # Determine the side of the line
                        current_side = self.check_side(bbox_center, (entry_line_start, entry_line_end))
                        intersect = LineIntersectionTest(bbox, line_coord).point_line_intersection_test()
                        # Logic for counting based on the crossing state
                        if state == "none":
                            # Check if the person is crossing the line
                            if current_side > self.entry_zone:  # Entering the "In" zone
                                self.tracked_ids[tracker_id]["state"] = "crossing_out"

                            elif current_side < self.exit_zone:  # Entering the "Out" zone
                                self.tracked_ids[tracker_id]["state"] = "crossing_in"

                        elif state == "crossing_in" and self.in_id[tracker_id]["count"] == 0:
                            if current_side > self.entry_zone and intersect:  # Successfully crossed into the "Out" zone
                                self.person_count_in += 1
                                # self.in_id[tracker_id]["count"] = self.in_id[tracker_id]["count"] + 1
                                self.in_id[tracker_id]["count"] = self.in_id.get("count", 0) + 1
                                self.tracked_ids[tracker_id]["state"] = "none"  # Reset state for new counts

                        elif state == "crossing_out" and self.out_id[tracker_id]["count"] == 0:
                            if current_side < self.exit_zone and intersect:  # Successfully crossed into the "In" zone
                                self.person_count_out += 1
                                # self.out_id[tracker_id]["count"] = self.out_id[tracker_id]["count"] + 1
                                self.out_id[tracker_id]["count"] = self.out_id.get("count", 0) + 1
                                self.tracked_ids[tracker_id]["state"] = "none"  # Reset state for new counts

                # Display counts on frame
                cv2.putText(frame, f'In: {self.person_count_in}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Out: {self.person_count_out}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                return frame
            else:
                cv2.putText(frame, f'In: {self.person_count_in}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Out: {self.person_count_out}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            2)
                return frame

        except Exception as er:
            print(er)
