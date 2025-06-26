# src/utils/realtime_progress.py - REAL-TIME PROGRESS MONITORING COMPONENTS

import streamlit as st
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Processing stages for progress tracking."""
    UPLOAD = "upload"
    OCR_EXTRACTION = "ocr_extraction"
    TEXT_SANITIZATION = "text_sanitization"
    CHUNKING = "chunking"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    STANDARDIZATION = "standardization"
    INFERENCE = "inference"
    NEO4J_STORAGE = "neo4j_storage"
    VECTOR_STORAGE = "vector_storage"
    COMPLETION = "completion"


@dataclass
class ProgressUpdate:
    """Data structure for progress updates."""
    job_id: str
    file_name: str
    stage: ProcessingStage
    progress_percent: float
    status: str
    message: str
    timestamp: str
    details: Optional[Dict] = None
    error: Optional[str] = None


class ProgressTracker:
    """Thread-safe progress tracking for real-time updates."""

    def __init__(self):
        self._updates: Dict[str, List[ProgressUpdate]] = {}
        self._current_status: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []

    def register_callback(self, callback: Callable):
        """Register callback for progress updates."""
        self._callbacks.append(callback)

    def update_progress(self, update: ProgressUpdate):
        """Update progress for a job/file."""
        with self._lock:
            if update.job_id not in self._updates:
                self._updates[update.job_id] = []

            self._updates[update.job_id].append(update)

            # Update current status
            if update.job_id not in self._current_status:
                self._current_status[update.job_id] = {}

            self._current_status[update.job_id][update.file_name] = {
                'stage': update.stage.value,
                'progress': update.progress_percent,
                'status': update.status,
                'message': update.message,
                'timestamp': update.timestamp,
                'error': update.error
            }

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Get current progress for a job."""
        with self._lock:
            return self._current_status.get(job_id, {})

    def get_job_updates(self, job_id: str, since: Optional[str] = None) -> List[ProgressUpdate]:
        """Get all updates for a job, optionally since a timestamp."""
        with self._lock:
            updates = self._updates.get(job_id, [])

            if since:
                try:
                    since_dt = datetime.fromisoformat(since)
                    updates = [u for u in updates if datetime.fromisoformat(u.timestamp) > since_dt]
                except ValueError:
                    pass

            return updates


# Global progress tracker instance
_progress_tracker = ProgressTracker()


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance."""
    return _progress_tracker


class LiveProgressDisplay:
    """Streamlit component for live progress display."""

    def __init__(self, job_id: str, total_files: int):
        self.job_id = job_id
        self.total_files = total_files
        self.tracker = get_progress_tracker()

        # Create Streamlit containers
        self.main_container = st.container()
        self.setup_display()

    def setup_display(self):
        """Setup the progress display interface."""
        with self.main_container:
            st.subheader(f"📊 Live Progress - Job {self.job_id[:8]}...")

            # Overall progress section
            self.overall_container = st.container()
            with self.overall_container:
                st.markdown("### 🎯 Overall Progress")
                self.overall_progress = st.progress(0)
                self.overall_status = st.empty()
                self.overall_metrics = st.empty()

            # File-by-file progress section
            st.markdown("### 📄 File Processing Details")
            self.files_container = st.container()

            # Performance metrics section
            st.markdown("### 📈 Performance Metrics")
            self.metrics_container = st.container()

    def update_display(self):
        """Update the progress display with latest data."""
        try:
            progress_data = self.tracker.get_job_progress(self.job_id)

            if not progress_data:
                with self.overall_container:
                    self.overall_status.info("⏳ Waiting for processing to start...")
                return

            # Calculate overall progress
            total_progress = 0
            completed_files = 0
            processing_files = 0
            failed_files = 0

            file_statuses = []

            for file_name, file_data in progress_data.items():
                progress = file_data.get('progress', 0)
                status = file_data.get('status', 'unknown')

                total_progress += progress

                if progress >= 100:
                    completed_files += 1
                elif progress > 0:
                    processing_files += 1

                if status.lower() in ['failed', 'error']:
                    failed_files += 1

                file_statuses.append({
                    'name': file_name,
                    'progress': progress,
                    'status': status,
                    'stage': file_data.get('stage', 'unknown'),
                    'message': file_data.get('message', ''),
                    'error': file_data.get('error')
                })

            # Update overall progress
            overall_progress = total_progress / self.total_files if self.total_files > 0 else 0
            self.overall_progress.progress(min(overall_progress / 100, 1.0))

            # Update overall status
            with self.overall_status:
                if processing_files > 0:
                    st.info(
                        f"🔄 Processing: {processing_files} files | Completed: {completed_files} | Failed: {failed_files}")
                elif completed_files == self.total_files:
                    st.success(f"✅ All files completed! Total: {completed_files}")
                else:
                    st.warning(f"⚠️ Processing issues - Completed: {completed_files} | Failed: {failed_files}")

            # Update overall metrics
            with self.overall_metrics:
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Progress", f"{overall_progress:.1f}%")
                with metric_cols[1]:
                    st.metric("Completed", completed_files)
                with metric_cols[2]:
                    st.metric("Processing", processing_files)
                with metric_cols[3]:
                    st.metric("Failed", failed_files)

            # Update file details
            with self.files_container:
                for file_status in file_statuses:
                    self._display_file_progress(file_status)

            # Update performance metrics
            self._update_performance_metrics(progress_data)

        except Exception as e:
            logger.error(f"Error updating progress display: {e}")
            st.error(f"❌ Progress display error: {e}")

    def _display_file_progress(self, file_status: Dict):
        """Display progress for individual file."""
        file_name = file_status['name']
        progress = file_status['progress']
        status = file_status['status']
        stage = file_status['stage']
        message = file_status['message']
        error = file_status['error']

        # Status icon
        status_icons = {
            'processing': '🔄',
            'completed': '✅',
            'success': '✅',
            'failed': '❌',
            'error': '❌',
            'waiting': '⏳',
            'cached': '🎯'
        }

        icon = status_icons.get(status.lower(), '❓')

        # Create expandable section for each file
        with st.expander(f"{icon} {file_name} - {stage.title()} ({progress:.1f}%)",
                         expanded=progress > 0 and progress < 100):
            # File progress bar
            st.progress(min(progress / 100, 1.0))

            # Status information
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Status:** {status.title()}")
                st.write(f"**Stage:** {stage.replace('_', ' ').title()}")
            with col2:
                st.write(f"**Progress:** {progress:.1f}%")
                if message:
                    st.write(f"**Message:** {message}")

            # Error display
            if error:
                st.error(f"**Error:** {error}")

            # Stage progress indicator
            self._display_stage_progress(stage, progress)

    def _display_stage_progress(self, current_stage: str, progress: float):
        """Display visual stage progress indicator."""
        stages = [
            ('upload', 'Upload'),
            ('ocr_extraction', 'OCR'),
            ('text_sanitization', 'Sanitization'),
            ('chunking', 'Chunking'),
            ('knowledge_extraction', 'Knowledge Graph'),
            ('standardization', 'Standardization'),
            ('inference', 'Inference'),
            ('neo4j_storage', 'Neo4j Storage'),
            ('vector_storage', 'Vector Storage'),
            ('completion', 'Completion')
        ]

        # Create stage indicators
        stage_cols = st.columns(len(stages))

        for i, (stage_key, stage_name) in enumerate(stages):
            with stage_cols[i]:
                if stage_key == current_stage:
                    if progress >= 100:
                        st.markdown(f"✅ **{stage_name}**")
                    else:
                        st.markdown(f"🔄 **{stage_name}**")
                elif stages.index((current_stage, None)) > i:
                    st.markdown(f"✅ {stage_name}")
                else:
                    st.markdown(f"⏳ {stage_name}")

    def _update_performance_metrics(self, progress_data: Dict):
        """Update performance metrics display."""
        with self.metrics_container:
            # Calculate performance statistics
            total_time = 0
            completed_count = 0
            avg_progress = 0

            stage_counts = {}

            for file_data in progress_data.values():
                if file_data.get('progress', 0) >= 100:
                    completed_count += 1

                avg_progress += file_data.get('progress', 0)

                stage = file_data.get('stage', 'unknown')
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

            avg_progress = avg_progress / len(progress_data) if progress_data else 0

            # Display metrics
            perf_cols = st.columns(3)

            with perf_cols[0]:
                st.metric("Average Progress", f"{avg_progress:.1f}%")
                st.metric("Completion Rate", f"{completed_count}/{len(progress_data)}")

            with perf_cols[1]:
                st.write("**Stage Distribution:**")
                for stage, count in stage_counts.items():
                    st.write(f"- {stage.replace('_', ' ').title()}: {count}")

            with perf_cols[2]:
                # Processing speed (if available)
                st.write("**Performance:**")
                if completed_count > 0:
                    st.write(f"✅ Files Completed: {completed_count}")
                processing_count = len(progress_data) - completed_count
                if processing_count > 0:
                    st.write(f"🔄 Files Processing: {processing_count}")


class AutoRefreshMonitor:
    """Auto-refreshing monitor for real-time updates."""

    def __init__(self, job_id: str, refresh_interval: int = 2):
        self.job_id = job_id
        self.refresh_interval = refresh_interval
        self.display = None
        self.is_active = True

    def start_monitoring(self, total_files: int):
        """Start the monitoring display."""
        # Initialize display
        self.display = LiveProgressDisplay(self.job_id, total_files)

        # Create auto-refresh mechanism
        self._setup_auto_refresh()

    def _setup_auto_refresh(self):
        """Setup auto-refresh for real-time updates."""
        # Create refresh control container
        refresh_container = st.container()

        with refresh_container:
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"**Auto-refresh:** Every {self.refresh_interval} seconds")

            with col2:
                if st.button("🔄 Manual Refresh"):
                    self._manual_refresh()

            with col3:
                if st.button("⏹️ Stop Auto-Refresh"):
                    self.is_active = False
                    st.info("Auto-refresh stopped")

        # Auto-refresh loop
        if self.is_active:
            self._refresh_loop()

    def _manual_refresh(self):
        """Manual refresh trigger."""
        if self.display:
            self.display.update_display()
        st.rerun()

    def _refresh_loop(self):
        """Auto-refresh loop."""
        # Update display
        if self.display:
            self.display.update_display()

        # Check if job is still active
        from src.utils.processing_pipeline import is_job_running

        if is_job_running(self.job_id):
            # Schedule next refresh
            time.sleep(self.refresh_interval)
            st.rerun()
        else:
            # Job completed
            st.success("✅ Job completed! Auto-refresh stopped.")
            self.is_active = False


class ProgressAnalytics:
    """Analytics and insights for progress data."""

    @staticmethod
    def generate_job_summary(job_id: str) -> Dict[str, Any]:
        """Generate comprehensive job summary."""
        tracker = get_progress_tracker()
        updates = tracker.get_job_updates(job_id)

        if not updates:
            return {"error": "No progress data found"}

        # Calculate statistics
        total_files = len(set(u.file_name for u in updates))
        completed_files = len(set(u.file_name for u in updates if u.progress_percent >= 100))
        failed_files = len(set(u.file_name for u in updates if u.status.lower() in ['failed', 'error']))

        # Time analysis
        start_time = min(datetime.fromisoformat(u.timestamp) for u in updates)
        end_time = max(datetime.fromisoformat(u.timestamp) for u in updates)
        total_duration = (end_time - start_time).total_seconds()

        # Stage analysis
        stage_durations = {}
        for file_name in set(u.file_name for u in updates):
            file_updates = [u for u in updates if u.file_name == file_name]

            for i, update in enumerate(file_updates[:-1]):
                next_update = file_updates[i + 1]
                stage = update.stage.value

                duration = (
                        datetime.fromisoformat(next_update.timestamp) -
                        datetime.fromisoformat(update.timestamp)
                ).total_seconds()

                if stage not in stage_durations:
                    stage_durations[stage] = []
                stage_durations[stage].append(duration)

        # Calculate average stage durations
        avg_stage_durations = {
            stage: sum(durations) / len(durations)
            for stage, durations in stage_durations.items()
            if durations
        }

        return {
            "job_id": job_id,
            "total_files": total_files,
            "completed_files": completed_files,
            "failed_files": failed_files,
            "success_rate": completed_files / total_files if total_files > 0 else 0,
            "total_duration_seconds": total_duration,
            "avg_stage_durations": avg_stage_durations,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }

    @staticmethod
    def show_analytics_dashboard(job_id: str):
        """Display comprehensive analytics dashboard."""
        summary = ProgressAnalytics.generate_job_summary(job_id)

        if "error" in summary:
            st.error(f"❌ {summary['error']}")
            return

        st.subheader("📊 Job Analytics Dashboard")

        # Key metrics
        metrics_cols = st.columns(4)

        with metrics_cols[0]:
            st.metric("Total Files", summary['total_files'])
        with metrics_cols[1]:
            st.metric("Success Rate", f"{summary['success_rate']:.1%}")
        with metrics_cols[2]:
            st.metric("Duration", f"{summary['total_duration_seconds']:.1f}s")
        with metrics_cols[3]:
            files_per_second = summary['total_files'] / summary['total_duration_seconds'] if summary[
                                                                                                 'total_duration_seconds'] > 0 else 0
            st.metric("Throughput", f"{files_per_second:.2f} files/s")

        # Stage performance
        if summary['avg_stage_durations']:
            st.subheader("⏱️ Stage Performance")

            import pandas as pd

            stage_data = []
            for stage, duration in summary['avg_stage_durations'].items():
                stage_data.append({
                    'Stage': stage.replace('_', ' ').title(),
                    'Avg Duration (s)': f"{duration:.2f}",
                    'Percentage': f"{duration / summary['total_duration_seconds'] * 100:.1f}%"
                })

            stage_df = pd.DataFrame(stage_data)
            st.dataframe(stage_df, use_container_width=True, hide_index=True)

        # Timeline visualization
        st.subheader("📈 Processing Timeline")
        ProgressAnalytics._show_timeline_chart(job_id)

    @staticmethod
    def _show_timeline_chart(job_id: str):
        """Show processing timeline chart."""
        tracker = get_progress_tracker()
        updates = tracker.get_job_updates(job_id)

        if not updates:
            st.info("No timeline data available")
            return

        try:
            import pandas as pd
            import plotly.express as px

            # Prepare data for timeline
            timeline_data = []
            for update in updates:
                timeline_data.append({
                    'File': update.file_name,
                    'Stage': update.stage.value.replace('_', ' ').title(),
                    'Progress': update.progress_percent,
                    'Timestamp': datetime.fromisoformat(update.timestamp),
                    'Status': update.status
                })

            df = pd.DataFrame(timeline_data)

            # Create timeline chart
            fig = px.line(
                df,
                x='Timestamp',
                y='Progress',
                color='File',
                title='Processing Progress Over Time',
                labels={'Progress': 'Progress (%)', 'Timestamp': 'Time'}
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.info("Install plotly for timeline visualization: pip install plotly")
        except Exception as e:
            st.error(f"Timeline chart error: {e}")


# Integration functions for existing pipeline

def create_progress_monitor(job_id: str, total_files: int) -> AutoRefreshMonitor:
    """Create and start a progress monitor for a job."""
    monitor = AutoRefreshMonitor(job_id)
    monitor.start_monitoring(total_files)
    return monitor


def update_file_progress(job_id: str, file_name: str, stage: ProcessingStage,
                         progress: float, status: str, message: str = "",
                         details: Dict = None, error: str = None):
    """Update progress for a specific file in a job."""
    update = ProgressUpdate(
        job_id=job_id,
        file_name=file_name,
        stage=stage,
        progress_percent=progress,
        status=status,
        message=message,
        timestamp=datetime.now().isoformat(),
        details=details,
        error=error
    )

    tracker = get_progress_tracker()
    tracker.update_progress(update)


def show_job_analytics(job_id: str):
    """Show analytics dashboard for a completed job."""
    ProgressAnalytics.show_analytics_dashboard(job_id)


# Example usage functions

def demo_progress_tracking():
    """Demo function showing how to use progress tracking."""
    st.title("🔄 Real-Time Progress Tracking Demo")

    if st.button("Start Demo Job"):
        # Simulate a processing job
        job_id = "demo_job_123"
        files = ["document1.pdf", "document2.txt", "document3.png"]

        # Create progress monitor
        monitor = create_progress_monitor(job_id, len(files))

        # Simulate processing stages
        for i, file_name in enumerate(files):
            stages = [
                (ProcessingStage.UPLOAD, 10, "Uploaded"),
                (ProcessingStage.OCR_EXTRACTION, 30, "Extracting text"),
                (ProcessingStage.CHUNKING, 50, "Creating chunks"),
                (ProcessingStage.KNOWLEDGE_EXTRACTION, 80, "Extracting knowledge"),
                (ProcessingStage.NEO4J_STORAGE, 95, "Storing in Neo4j"),
                (ProcessingStage.COMPLETION, 100, "Completed")
            ]

            for stage, progress, message in stages:
                update_file_progress(
                    job_id=job_id,
                    file_name=file_name,
                    stage=stage,
                    progress=progress,
                    status="processing" if progress < 100 else "completed",
                    message=message
                )
                time.sleep(0.5)  # Simulate processing time


if __name__ == "__main__":
    # Demo the progress tracking system
    demo_progress_tracking()