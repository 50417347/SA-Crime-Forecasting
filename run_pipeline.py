import os
import subprocess
from datetime import datetime


def run_step(command, step_name, log_file):
    print(f"\nRunning: {step_name}")
    log_file.write(f"\n[{datetime.now()}] Running: {step_name}\n")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    log_file.write(result.stdout)
    log_file.write(result.stderr)

    if result.returncode != 0:
        print(f"Failed: {step_name}")
        print(result.stderr)
        raise RuntimeError(f"{step_name} failed")
    else:
        print(f"Completed: {step_name}")


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    log_name = datetime.now().strftime("logs/pipeline_%Y%m%d_%H%M%S.log")

    with open(log_name, "w", encoding="utf-8") as log_file:
        log_file.write(f"Pipeline started at {datetime.now()}\n")

        run_step("python clean_data.py", "Clean Data", log_file)
        run_step("python train_model.py", "Train Model", log_file)
        run_step("python report_generator.py", "Generate Report", log_file)
        run_step("python pdf_report.py", "Generate PDF Report", log_file)
        run_step("python hotspot_map.py", "Generate Hotspot Map", log_file)

        log_file.write(f"\nPipeline finished at {datetime.now()}\n")

    print("\nPipeline completed successfully.")
    print(f"Log saved to: {log_name}")


if __name__ == "__main__":
    main()