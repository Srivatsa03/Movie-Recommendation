# online/metrics.py
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

MODEL_VERSION = "v1.0"
PIPELINE_VERSION = "online_eval_1.0"
TELEMETRY_VERSION = "v1.0"


class MetricsComputer:
    def __init__(self, telemetry_dir="telemetry", output_dir="evaluation_results"):
        self.telemetry_dir = telemetry_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.recommendations, self.watches, self.ratings = self._load_telemetry()

    def _load_telemetry(self):
        def load_json(file_name):
            path = os.path.join(self.telemetry_dir, file_name)
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
            return {}

        return (
            load_json("recommendations.json"),
            load_json("watches.json"),
            load_json("ratings.json"),
        )

    def _filter_by_time_window(self, data_dict, time_window, key_func):
        if not time_window:
            return data_dict
        cutoff = datetime.now() - timedelta(hours=time_window)
        filtered = {}
        for uid, events in data_dict.items():
            valid_events = [event for event in events if key_func(event) >= cutoff]
            if valid_events:
                filtered[uid] = valid_events
        return filtered

    def compute_ctr(self, time_window=None):
        recommendations = self._filter_by_time_window(
            self.recommendations,
            time_window,
            lambda e: datetime.fromisoformat(e["timestamp"]),
        )
        total_recs = 0
        total_clicks = 0

        for uid, rec_list in recommendations.items():
            for event in rec_list:
                recs = event["recommendations"]
                total_recs += len(recs)
                if uid in self.watches:
                    total_clicks += sum(
                        1 for movie in recs if movie in self.watches[uid]
                    )

        return (total_clicks / total_recs) * 100 if total_recs else 0

    def compute_wtr(self, min_minutes=5, time_window=None):
        recommendations = self._filter_by_time_window(
            self.recommendations,
            time_window,
            lambda e: datetime.fromisoformat(e["timestamp"]),
        )
        total_recs = 0
        total_significant = 0

        for uid, rec_list in recommendations.items():
            for event in rec_list:
                recs = event["recommendations"]
                total_recs += len(recs)
                if uid in self.watches:
                    for movie in recs:
                        if movie in self.watches[uid]:
                            max_min = max(
                                ev["minute"] for ev in self.watches[uid][movie]
                            )
                            if max_min >= min_minutes:
                                total_significant += 1

        return (total_significant / total_recs) * 100 if total_recs else 0

    def compute_rating_quality(self, time_window=None):
        recommendations = self._filter_by_time_window(
            self.recommendations,
            time_window,
            lambda e: datetime.fromisoformat(e["timestamp"]),
        )
        total_ratings = 0
        rating_sum = 0

        for uid, rec_list in recommendations.items():
            for event in rec_list:
                recs = event["recommendations"]
                if uid in self.ratings:
                    for movie in recs:
                        if movie in self.ratings[uid]:
                            rating_sum += self.ratings[uid][movie]["rating"]
                            total_ratings += 1

        return (rating_sum / total_ratings) if total_ratings else 0

    def generate_report(self, time_windows=[24, 72, 168]):
        self.recommendations, self.watches, self.ratings = self._load_telemetry()

        summary = {"generated_at": datetime.now().isoformat(), "metrics": {}}

        for hours in time_windows:
            summary["metrics"][f"{hours}h"] = {
                "ctr": self.compute_ctr(hours),
                "wtr": self.compute_wtr(5, hours),
                "avg_rating": self.compute_rating_quality(hours),
            }

        summary["metrics"]["all_time"] = {
            "ctr": self.compute_ctr(),
            "wtr": self.compute_wtr(),
            "avg_rating": self.compute_rating_quality(),
        }

        print(
            f"Provenance | Model Version: {MODEL_VERSION}, Pipeline: {PIPELINE_VERSION}, Telemetry Version: {TELEMETRY_VERSION}"
        )

        output_path = os.path.join(self.output_dir, "online_evaluation_report.json")
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def generate_visualizations(self):
        """Generate visualizations of metrics over time

        Returns:
            list: Paths to generated visualization files
        """
        # Reload telemetry data
        self.recommendations, self.watches, self.ratings = self._load_telemetry()

        # Generate timestamps for the last 14 days in 24-hour intervals
        end_time = datetime.now()
        start_time = end_time - timedelta(days=14)
        intervals = []

        current = start_time
        while current < end_time:
            next_time = current + timedelta(hours=24)
            intervals.append((current, next_time))
            current = next_time

        # Prepare data for visualization
        timestamps = []
        ctr_values = []
        wtr_values = []
        rating_values = []

        for interval_start, interval_end in intervals:
            # Filter recommendations in this interval
            window_recommendations = {}
            for user_id, user_recs in self.recommendations.items():
                window_recommendations[user_id] = []
                for rec in user_recs:
                    rec_time = datetime.fromisoformat(rec["timestamp"])
                    if interval_start <= rec_time < interval_end:
                        window_recommendations[user_id].append(rec)

            # Compute metrics for this interval
            ctr = self.compute_ctr_for_recs(window_recommendations)
            wtr = self.compute_wtr_for_recs(window_recommendations)
            avg_rating = self.compute_rating_for_recs(window_recommendations)

            # Store values
            timestamps.append(interval_start.strftime("%m-%d"))
            ctr_values.append(ctr)
            wtr_values.append(wtr)
            rating_values.append(avg_rating)

        # Create directory for visualizations
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Generate plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots = []

        # CTR Plot
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, ctr_values, marker="o", linestyle="-", color="blue")
        plt.title("Click-Through Rate Over Time")
        plt.xlabel("Date")
        plt.ylabel("CTR (%)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()

        ctr_plot = os.path.join(viz_dir, f"ctr_{timestamp}.png")
        plt.savefig(ctr_plot)
        plt.close()
        plots.append(ctr_plot)

        # WTR Plot
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, wtr_values, marker="o", linestyle="-", color="green")
        plt.title("Watch-Through Rate Over Time")
        plt.xlabel("Date")
        plt.ylabel("WTR (%)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()

        wtr_plot = os.path.join(viz_dir, f"wtr_{timestamp}.png")
        plt.savefig(wtr_plot)
        plt.close()
        plots.append(wtr_plot)

        # Average Rating Plot
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, rating_values, marker="o", linestyle="-", color="red")
        plt.title("Average Rating of Recommended Movies")
        plt.xlabel("Date")
        plt.ylabel("Average Rating")
        plt.ylim(0, 5)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()

        rating_plot = os.path.join(viz_dir, f"ratings_{timestamp}.png")
        plt.savefig(rating_plot)
        plt.close()
        plots.append(rating_plot)

        # Dashboard (combined metrics)
        plt.figure(figsize=(15, 10))

        # CTR subplot
        plt.subplot(2, 2, 1)
        plt.plot(timestamps, ctr_values, marker="o", linestyle="-", color="blue")
        plt.title("Click-Through Rate")
        plt.xlabel("Date")
        plt.ylabel("CTR (%)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)

        # WTR subplot
        plt.subplot(2, 2, 2)
        plt.plot(timestamps, wtr_values, marker="o", linestyle="-", color="green")
        plt.title("Watch-Through Rate")
        plt.xlabel("Date")
        plt.ylabel("WTR (%)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)

        # Rating subplot
        plt.subplot(2, 2, 3)
        plt.plot(timestamps, rating_values, marker="o", linestyle="-", color="red")
        plt.title("Average Rating")
        plt.xlabel("Date")
        plt.ylabel("Rating")
        plt.ylim(0, 5)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)

        plt.tight_layout()

        dashboard_plot = os.path.join(viz_dir, f"dashboard_{timestamp}.png")
        plt.savefig(dashboard_plot)
        plt.close()
        plots.append(dashboard_plot)

        print(f"Generated {len(plots)} visualization plots")
        return plots

    def compute_ctr_for_recs(self, recommendations):
        """Helper method to compute CTR for a specific set of recommendations"""
        clicks = 0
        total_recommendations = 0

        for user_id, user_recommendations in recommendations.items():
            for rec_event in user_recommendations:
                recommended_movies = rec_event["recommendations"]
                total_recommendations += len(recommended_movies)

                if user_id in self.watches:
                    for movie_id in recommended_movies:
                        if movie_id in self.watches[user_id]:
                            clicks += 1

        if total_recommendations == 0:
            return 0

        return (clicks / total_recommendations) * 100

    def compute_wtr_for_recs(self, recommendations, min_watch_minutes=5):
        """Helper method to compute WTR for a specific set of recommendations"""
        significant_watches = 0
        total_recommendations = 0

        for user_id, user_recommendations in recommendations.items():
            for rec_event in user_recommendations:
                recommended_movies = rec_event["recommendations"]
                total_recommendations += len(recommended_movies)

                if user_id in self.watches:
                    for movie_id in recommended_movies:
                        if movie_id in self.watches[user_id]:
                            max_minute = max(
                                event["minute"]
                                for event in self.watches[user_id][movie_id]
                            )
                            if max_minute >= min_watch_minutes:
                                significant_watches += 1

        if total_recommendations == 0:
            return 0

        return (significant_watches / total_recommendations) * 100

    def compute_rating_for_recs(self, recommendations):
        """Helper method to compute average rating for a specific set of recommendations"""
        total_ratings = 0
        sum_ratings = 0

        for user_id, user_recommendations in recommendations.items():
            for rec_event in user_recommendations:
                recommended_movies = rec_event["recommendations"]

                if user_id in self.ratings:
                    for movie_id in recommended_movies:
                        if movie_id in self.ratings[user_id]:
                            sum_ratings += self.ratings[user_id][movie_id]["rating"]
                            total_ratings += 1

        if total_ratings == 0:
            return 0

        return sum_ratings / total_ratings
