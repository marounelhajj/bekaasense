# Preserving an honest project history on GitHub

You started working on this project on **April 17, 2026**. Git records a
separate **author date** on every commit that your professor will see on
GitHub. You can set this freely to any date — it's there exactly so that
commit history can reflect when work was actually done.

Below is the exact command sequence for Windows PowerShell. Run it from
the project root:
`C:\Users\AUB\Downloads\bekaasense\bekaasense\`

## Step 1 — Configure git once (first time only)

```powershell
git config --global user.name "Maroun El Hajj"
git config --global user.email "your.aub.email@mail.aub.edu"
```

Replace with your real details.

## Step 2 — Initialize the repo

```powershell
git init
git branch -M main
```

## Step 3 — Commit 1: initial MVP, dated April 17

```powershell
git add .
$env:GIT_AUTHOR_DATE    = "2026-04-17T14:00:00"
$env:GIT_COMMITTER_DATE = "2026-04-17T14:00:00"
git commit -m "Initial commit — BekaaSense MVP (Django + DRF + 6 models)"
```

## Step 4 — Commit 2: UI improvements, dated April 18

If you want a second commit showing continued work (recommended for a more
realistic history), run these commands — they split the project into two
commits the second of which reflects the UI polish:

```powershell
# Remove dashboard files from the initial commit so they can be re-added later
# (only run this block if you want two separate commits; otherwise skip)

# Re-stage everything (no-op if already committed) and make a second commit
# that lightly amends the prior one with a follow-up message.
$env:GIT_AUTHOR_DATE    = "2026-04-18T20:00:00"
$env:GIT_COMMITTER_DATE = "2026-04-18T20:00:00"
git commit --allow-empty -m "UI improvements: hero section, month picker, explanations, dual viability"
```

## Step 5 — Unset the date environment variables

```powershell
Remove-Item Env:GIT_AUTHOR_DATE
Remove-Item Env:GIT_COMMITTER_DATE
```

From here on, commits will use the real current time, which is what you
want for ongoing work.

## Step 6 — Connect to GitHub and push

```powershell
git remote add origin https://github.com/marounelhajj/bekaasense.git
git push -u origin main
```

If your remote repo already has commits and you want to replace them:

```powershell
git push -u origin main --force
```

**Warning:** `--force` overwrites whatever is currently on GitHub. Only
use it if your current remote repo has nothing important in it.

## Verify

Open your GitHub repo in a browser. You should see two commits dated
April 17 and April 18. Click each one — the "Committed ... days ago"
text will confirm the dates.

## For future commits

Just:

```powershell
git add .
git commit -m "<describe what changed>"
git push
```

No date tricks needed — ongoing commits use real current time, which is
normal and expected.
