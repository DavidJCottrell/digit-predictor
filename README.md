## Setup (WSL and Ubuntu)

Clone this repository: `https://github.com/DavidJCottrell/digit-predictor.git`.

Create and source a virtual environment in the repo directory:

`python3 -m venv ./venv`

`source ./venv/bin/activate`

Install Dependencies:

`pip install -r requirements.txt`

## Run Predictions

run `python3 predict.py` and open `test_number.png` in paint - the number will be predicted each time you save the file.

![alt text](https://github.com/DavidJCottrell/digit-predictor/blob/main/demo.png)
