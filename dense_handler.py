import os
import shutil
import subprocess
import time
from pathlib import Path

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.event_type import EventType
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.app_constant import AppConstants


class DenseEventHandler(FLComponent):
    """Debug-instrumented DENSE handler: logs every path and directory creation."""

    # adjust these to your actual script locations if needed
    DENSE_GENERATE = (
        "/path/to/root/repo/prostate_2D/"
        "job_configs/picai_fedsemi/app/dense_generate.py"
    )
    DENSE_DISTILL = (
        "/path/to/root/repo/prostate_2D/"
        "job_configs/picai_fedsemi/app/dense_distill.py"
    )

    def _run_dir(self, fl_ctx: FLContext) -> Path:
        return Path(
            fl_ctx.get_engine()
            .get_workspace()
            .get_run_dir(fl_ctx.get_job_id())
        )

    def _persistor_file(self, run_dir: Path) -> Path:
        # PTFileModelPersistor writes this file each round
        return run_dir / "app_server" / "FL_global_model.pt"

    def _dense_file(self, run_dir: Path, rnd: int | str) -> Path:
        # dense_generate.py expects this relative layout
        return run_dir / "app_server" / "checkpoints" / f"round-{rnd}.pt"

    def _wait_for_file(self, path: Path, timeout: float, fl_ctx: FLContext) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if path.exists():
                return True
            time.sleep(0.2)
        return False


    def _run_dense(self, tag: str | int, cwd: Path, fl_ctx: FLContext):
        if not isinstance(tag, int):
            self.log_info(
                fl_ctx,
                "[DENSE] END_RUN tag detected – generator and distiller skipped",
            )
            return

        self.log_info(fl_ctx, f"[DEBUG] Launching DENSE round {tag}")

        subprocess.run(
            ["python", self.DENSE_GENERATE, "--round", str(tag)],
            cwd=cwd,
            check=True,
        )

        subprocess.run(
            ["python", self.DENSE_DISTILL, "--round", str(tag)],
            cwd=cwd,
            check=True,
        )

        run_dir = cwd / "simulate_job"
        distilled   = run_dir / "app_server" / "checkpoints" / f"distilled_round-{tag}.pt"
        global_ckpt = run_dir / "app_server" / "FL_global_model.pt"

        if distilled.exists():
            shutil.copyfile(distilled, global_ckpt)
            self.log_info(
                fl_ctx,
                f"[DENSE] Replaced global model with distilled weights for round {tag}",
            )
        else:
            self.log_warning(
                fl_ctx,
                f"[DENSE] Expected {distilled} but it was missing – keeping old global model",
            )
        self.log_info(fl_ctx, f"DENSE completed for round {tag}")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"[DEBUG] handle_event: {event_type}")
        if event_type == AppEventType.ROUND_DONE:

            rnd = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
            if rnd is None:
                self.log_warning(fl_ctx, "[DEBUG] no round index, skipping")
                return

            run_dir = self._run_dir(fl_ctx)

            src = self._persistor_file(run_dir)
            if not self._wait_for_file(src, 10.0, fl_ctx):
                return
            dst = self._dense_file(run_dir, rnd)

            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            except Exception as e:
                self.log_error(fl_ctx, f"[DEBUG] copy failed: {e}")
                return

            try:
                # run both scripts with the server workspace as cwd
                self._run_dense(rnd, run_dir.parent, fl_ctx)
            except subprocess.CalledProcessError as e:
                self.log_error(
                    fl_ctx, f"DENSE failed in round {rnd}; rc={e.returncode}"
                )
        elif event_type == EventType.END_RUN:
            run_dir = self._run_dir(fl_ctx)
            src = self._persistor_file(run_dir)
            if not self._wait_for_file(src, 10.0, fl_ctx):
                return
            dst = self._dense_file(run_dir, "final")

            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            except Exception as e:
                self.log_error(fl_ctx, f"[DEBUG] END_RUN copy failed: {e}")
                return

            try:
                self._run_dense("final", run_dir.parent, fl_ctx)
            except subprocess.CalledProcessError as e:
                self.log_error(fl_ctx, f"END_RUN DENSE failed; rc={e.returncode}")
