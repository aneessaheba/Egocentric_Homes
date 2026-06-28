# Contributing to EgoLoop

Thank you for your interest in contributing!

## Recording guidelines

- Use a head-mounted or wrist-mounted camera
- Ensure hands are visible in at least 70% of frames
- Record in well-lit environments
- Avoid shaky footage — stabilise the camera if possible

## Clip naming

Name your clips using the pattern: `ActivityName_S##.mp4`

Examples:
- `WashingDishes_S01.mp4`
- `ChoppingVegetables_S12.mp4`
- `FoldingLaundry_S03.mp4`

## Submission checklist

- [ ] Video is at least 15 seconds long
- [ ] Hands visible in most frames
- [ ] File named correctly (`Activity_S##.mp4`)
- [ ] No personally identifiable information in frame
- [ ] Uploaded via the EgoLoop contributor portal

## Quality scoring

Submissions are auto-scored on:

| Criterion         | Weight |
|-------------------|--------|
| Blur / sharpness  | 30%    |
| Brightness        | 15%    |
| Contrast          | 15%    |
| Hand visibility   | 25%    |
| Motion coverage   | 15%    |

A minimum score of **60 / 100** is required for acceptance.

## Reporting issues

Open an issue on GitHub with:
- A short description of the problem
- The clip name (if applicable)
- Steps to reproduce

## Code contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-fix`)
3. Commit your changes with clear messages
4. Open a pull request describing what you changed and why

## Activity categories

Currently accepting clips in these categories:

- **Kitchen Activities**: washing, chopping, cooking, cleaning dishes
- **Laundry & Tidying**: folding, ironing, sorting, putting away
- **Cleaning & Organising**: mopping, sweeping, dusting, organising shelves

Check the active campaign on the EgoLoop portal for specific task prompts
and any restrictions (e.g. required props or environment).

## Payout structure

Payouts are per accepted clip and vary by campaign:

| Tier     | Acceptance Rate | Bonus per clip |
|----------|----------------|----------------|
| Bronze   | 60 – 74%       | Base rate       |
| Silver   | 75 – 89%       | +10%            |
| Gold     | 90%+           | +20%            |

Payouts are processed monthly. See the campaign page for the base rate.

## Data privacy guidelines

- Do **not** record in locations where other people appear without consent
- Do **not** include personally identifiable documents, screens, or text in frame
- Do **not** record financial or medical materials
- Clip metadata (timestamps, device info) is stripped on upload

Violations may result in clip rejection and account suspension.

## Testing your setup

Before recording full sessions, test your setup with a 10-second clip:

```bash
# Run the full pipeline on a test clip
python pipeline/run_pipeline.py

# Check the quality score
cat assets/processed/quality/MyClip_quality.json
```

Aim for a score above 70 to leave a comfortable buffer above the 60 threshold.

## After your clip is accepted

Once a clip passes the quality gate:

1. You will receive a notification via email
2. The clip is added to the dataset and counted toward your campaign total
3. Your contributor stats (acceptance rate, total clips) update within 24 hours
4. Payout for the clip is included in the next monthly cycle

## Syncing with the main branch

If your fork falls behind `main`:

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

Always create a new feature branch from the latest `main` to avoid
merge conflicts.

## Recording equipment tips

**Lighting**
- Use diffuse, even lighting to avoid harsh shadows on your hands
- Avoid backlighting (camera pointing at a window)
- A ring light or two desk lamps work well for kitchen setups

**Camera mounting**
- Head-mount: GoPro chest harness or head strap gives stable first-person view
- Wrist-mount: captures hand interactions more closely but may be shakier
- Tripod: acceptable for stationary tasks (washing, chopping) if overhead angle

## Common rejection reasons

Based on historical submissions, these are the most common reasons clips
are rejected:

1. **Blur** (score < 20/30): shaky camera or low light
2. **Hand visibility** (score < 10/25): hands frequently out of frame
3. **Brightness** (score < 5/15): too dark or overexposed environment
4. **Duration too short**: clip under 10 seconds after trimming silence

Reviewing your clip in the portal before submitting catches most issues.
