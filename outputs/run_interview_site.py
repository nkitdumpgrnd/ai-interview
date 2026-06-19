import os
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    app_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(app_dir))
    os.chdir(app_dir)
    uvicorn.run("resume_interview_api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
