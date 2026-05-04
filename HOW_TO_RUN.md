# HOW TO RUN — MLOps Lab CIE
## Complete Step-by-Step Guide (VSCode + GitHub)

---

## PART A — GitHub Repository Setup

### Step 1: Login to GitHub
1. Open your browser → go to https://github.com/login
2. Login with the **same GitHub ID** you submitted to faculty

---

### Step 2: Create a Personal Access Token (PAT)
1. Go to https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Name: `MLOps-CIE`
4. Expiration: **30 days**
5. Under **"Select scopes"**, check ✅ **repo**
6. Click **"Generate token"**
7. ⚠️ **Copy the token NOW** (starts with `ghp_`) — GitHub won't show it again!

---

### Step 3: Create the GitHub Repository
1. Go to https://github.com/new
2. Fill in:
   - **Repository name:** `Internals_Basics`  ← exact spelling (capital I, capital B)
   - **Description:** `MLOps Lab CIE Submission`
   - **Visibility:** ✅ Public
   - **Initialize this repository:** ✅ Add a README file
3. Click **"Create repository"**

---

## PART B — Open Project in VSCode

### Step 4: Open VSCode Terminal
- Open VSCode
- Press **Ctrl + `** (backtick) to open the integrated terminal
- OR go to **Terminal → New Terminal**

---

### Step 5: Clone the Repository
In the VSCode terminal, run:

```bash
git clone https://github.com/YOUR_USERNAME/Internals_Basics.git
```
Replace `YOUR_USERNAME` with your actual GitHub username.

When prompted:
- **Username:** your GitHub username
- **Password:** paste your PAT token (NOT your GitHub password)

Then enter the folder:
```bash
cd Internals_Basics
```

---

### Step 6: Copy the Project Files
Place the `MLOPs_Lab_CIE/` folder (from this zip) inside the cloned `Internals_Basics/` folder.

Your structure should look like:
```
Internals_Basics/
├── README.md
└── MLOPs_Lab_CIE/
    ├── data/
    ├── src/
    ├── models/
    ├── logs/
    ├── results/
    ├── requirements.txt
    └── .gitignore
```

---

## PART C — Python Setup in VSCode

### Step 7: Create Virtual Environment
In the VSCode terminal (make sure you're inside `Internals_Basics/MLOPs_Lab_CIE/`):

```bash
cd MLOPs_Lab_CIE
python -m venv venv
```

Activate the virtual environment:

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**On Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**On Mac/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

---

### Step 8: Install Dependencies
```bash
pip install -r requirements.txt
```
Wait 2–3 minutes for all packages to install.

---

## PART D — Run the Code

### Step 9: Train the Model
```bash
python src/train.py
```
This will:
- Load `data/training_data.csv`
- Train a RandomForest model
- Save model to `models/model.pkl`
- Save `results/step1_s1.json`

---

### Step 10: Batch Predict on New Data
```bash
python src/predict.py
```
This will:
- Load `data/new_data.csv`
- Run predictions using the trained model
- Save `results/step3_s5.json`

---

### Step 11: Evaluate the Model
```bash
python src/evaluate.py
```
This will:
- Evaluate model metrics (accuracy, precision, recall, F1)
- Run 5-fold cross-validation
- Save `results/step4_s8.json`

---

### Step 12: Run the FastAPI Server (optional)
First go back to the `Internals_Basics` root:
```bash
cd ..
```
Then run:
```bash
uvicorn MLOPs_Lab_CIE.src.api:app --reload
```
Open your browser → http://127.0.0.1:8000
API docs → http://127.0.0.1:8000/docs

---

## PART E — Push to GitHub

### Step 13: Push Your Work
Go back to the `Internals_Basics` root (the repo root):
```bash
cd ..   # if you're inside MLOPs_Lab_CIE
```
Then run:
```bash
git add .
git commit -m "MLOps CIE submission"
git push origin main
```

When prompted:
- **Username:** your GitHub username
- **Password:** paste your PAT token

**Tip — save credentials so you don't have to enter them every time:**
```bash
git config --global credential.helper store
```

---

### Step 14: Verify on GitHub
1. Go to `https://github.com/YOUR_USERNAME/Internals_Basics`
2. Click on the `MLOPs_Lab_CIE` folder
3. Confirm these are visible:
   - ✅ `data/` folder with CSV files
   - ✅ `src/` folder with Python files
   - ✅ `results/` folder with JSON files
   - ✅ `requirements.txt`
4. Click a results JSON — confirm it has actual values (not zeros)
5. Make sure repo is **Public**

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `git push` asks for password and fails | Use your PAT token (starts with `ghp_`), not your GitHub password |
| "remote: Permission denied" | Check PAT has `repo` scope |
| "failed to push, updates were rejected" | Run `git pull origin main --rebase` then push again |
| "repository not found" during clone | Check spelling of repo name and your username |
| Empty results folder | Run `python src/train.py` first, then the other scripts |
| Token expired or lost | Generate a new PAT at https://github.com/settings/tokens |

---

## Push After Every Task!
```bash
git add .
git commit -m "completed task 1"
git push origin main
```
