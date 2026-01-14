import streamlit as st
import pandas as pd
import time
import altair as alt
from collections import defaultdict

class AnalyticsTracker:
    def __init__(self):
        self.frame_counts = []  # List of (frame_idx, total_objects)
        self.class_counts = defaultdict(int)
        self.confidences = []
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def update(self, results):
        """Update stats with a new frame's results."""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed

        if not results:
            return

        result = results[0]  # Assuming single frame result
        
        # Count objects in this frame
        frame_obj_count = 0
        
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                
                self.class_counts[cls_name] += 1
                self.confidences.append(conf)
                frame_obj_count += 1

        self.frame_counts.append({"Frame": self.frame_count, "Count": frame_obj_count})

    def reset(self):
        """Reset stats for new video/stream."""
        self.__init__()

    def get_fps(self):
        return f"{self.fps:.2f}"

def render_dashboard(tracker: AnalyticsTracker):
    """Renders the analytics dashboard in Streamlit."""
    st.markdown("### 📊 Object Analytics Dashboard")
    
    # metrics row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Objects Detected", sum(tracker.class_counts.values()))
    with c2:
        avg_conf = sum(tracker.confidences) / len(tracker.confidences) if tracker.confidences else 0
        st.metric("Avg Confidence", f"{avg_conf:.2f}")
    with c3:
        st.metric("Real-time FPS", tracker.get_fps())

    # Charts row
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("**Objects per Class**")
        if tracker.class_counts:
            df_counts = pd.DataFrame(list(tracker.class_counts.items()), columns=["Class", "Count"])
            chart = alt.Chart(df_counts).mark_bar().encode(
                x=alt.X("Class", sort="-y"),
                y="Count",
                color="Class",
                tooltip=["Class", "Count"]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No detections yet.")

    with col_chart2:
        st.markdown("**Detection Confidence Distribution**")
        if tracker.confidences:
            df_conf = pd.DataFrame({"Confidence": tracker.confidences})
            chart = alt.Chart(df_conf).mark_bar().encode(
                x=alt.X("Confidence", bin=True),
                y="count()",
                tooltip=["count()"]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No data.")

    # Timeline for video
    if len(tracker.frame_counts) > 1:
        st.markdown("**Detections Over Time**")
        df_timeline = pd.DataFrame(tracker.frame_counts)
        chart_line = alt.Chart(df_timeline).mark_line().encode(
            x="Frame",
            y="Count",
            tooltip=["Frame", "Count"]
        ).properties(height=200).interactive()
        st.altair_chart(chart_line, use_container_width=True)
