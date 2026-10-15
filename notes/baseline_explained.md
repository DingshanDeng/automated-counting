# The Baseline Models, Explained from Scratch

This guide is for readers with little or no deep-learning or coding background. It explains:
- what each of our models (B0, B1, B4, and the planned B7) does
- why we tested each one
- how to read the results

The full numbers and figures are in [`baseline_results.md`](baseline_results.md). This page explains what they mean.

---

## 1. The task in one picture

Each image is a 640 × 640 pixel microscope photo, about 54 cells on average, from a **trypan blue** test. Trypan blue is a dye that only gets into **dead** cells, so dead cells look **blue** and live cells look **pale and clear**.

**Goal (for now):** given a new photo, say how many **cells** it contains in total.

Labs also care about how many are live and how many dead, which gives **viability**, the percentage of cells that are alive. We leave that for later. Most other cell datasets we want to use don't mark cells as live or dead, so we first solve plain counting. Telling kinds of cells apart (Live vs Dead here, or several kinds of cells in other datasets) is a later extension of the project.

Today a person counts the cells by eye, which is slow and varies from person to person. We want a computer to do it.

**What we learn from:** 500 photos where a person has already drawn a box around every cell and labelled it *Live* or *Dead*. That is 26,715 boxes in total: 25,691 Live and 1,024 Dead. These hand-drawn boxes are our **"truth"** (also called *labels* or *ground truth*). For now we only use how many boxes each photo has, not the Live/Dead label.

---

## 2. How we judge a model fairly

### Training, validation and test sets: the "homework, practice exam, final exam" idea

We split the 500 photos into three groups (the list is in `splits/splits.csv`) and never mix them up:

| Group | Images | Analogy | What it is used for |
|---|---|---|---|
| **Training** | 351 | Homework with the answer key | The model learns from these, and may look at the answers. |
| **Validation** | 74 | Practice exam | Used to check choices before the final exam, or to make small choices for models that train on the training set. |
| **Test** | 75 | The final exam | Used **once**, at the very end, to report the score. The model never sees these answers while learning. |

**Why bother?** A model can "memorize" its homework. If we graded it on photos it had already studied, it would look much better than it really is. The test set tells us how it would do on **new** photos.

### Leave-one-session-out: a harder exam

All 500 photos come from **5 imaging sessions** (5 runs at the microscope, 100 photos each). Each session has its own background colour and its own typical number of cells.

The danger: a model could learn "pink background → about 70 cells" instead of actually counting. That works on our photos, but fails at another lab.

So for models trained on our photos (such as YOLO) we also run a harder test, called **leave-one-session-out**:
1. Hide one whole session.
2. Learn from the other four.
3. Test on the hidden one.
4. Repeat 5 times, so every session gets hidden once.

This imitates "a photo from a microscope setup the model has never seen". We don't run it for the Cellpose baseline: Cellpose isn't trained on our photos, so it has nothing session-specific to learn.

### The scores, in plain words

| Score | Plain meaning | Good value |
|---|---|---|
| **MAE** (mean absolute error) | "On average, how many cells is the count off by?" Off by +3 on one photo and −1 on another gives MAE = 2. | **Lower is better**; 0 is perfect |
| **Within ±10%** | "On what share of photos is the total count within 10% of the truth?" For a 50-cell photo, 45–55 counts as a hit. | **Higher is better**; 100% is perfect |
| **R²** | "How well do the predicted counts follow the real counts from photo to photo?" 1 means perfectly, 0 means no better than always guessing the average, and below 0 means worse than that. | **Closer to 1 is better** |
| **Bias** | "Does the model count too many or too few, on average?" Negative means it undercounts. | **Closer to 0 is better** |
| **Precision** | "Of the cells the model **found**, how many are real cells?" Low precision means it marks things that aren't cells. | Higher is better |
| **Recall** | "Of the **real** cells, how many did the model find?" Low recall means it misses cells. | Higher is better |
| **F1** | One number that combines precision and recall | Higher is better |

**IoU** (overlap) decides whether a predicted box "found" a real cell. It measures how much the predicted box and the hand-drawn box overlap, from 0 (no overlap) to 1 (identical boxes). We usually count it as a match when IoU ≥ 0.5, meaning the boxes overlap by at least about half; we also report IoU ≥ 0.3 (see B4).

> **Why both counting scores and finding scores?** A model could miss 5 cells and wrongly add 5 other things, and still get the total exactly right. Precision and recall catch that.

---

## 3. Why are the models called B0, B1, B4, B7?

Our roadmap lists a **ladder** of eight possible baselines, B0 to B7, from the simplest to the most capable. Each rung answers the question: **how much does a smarter method actually add?**

| | Model | One-line idea | Deep learning? | Finds individual cells? | Status |
|---|---|---|---|---|---|
| **B0** | Average guess | Always guess the average count | No | No | ✅ Run |
| **B1** | Session average | Guess the average count of that photo's session | No | No | ✅ Run |
| B2 | Image statistics + formula | Predict the count from overall photo measurements (how "busy" or colourful it is) | No | No | Not run |
| B3 | Classic image processing | Hand-written rules: darken the background, threshold, split touching blobs | No | Yes | Not run |
| **B4** | Cellpose-SAM | A ready-made deep-learning "cell finder", used as downloaded | Yes (pre-trained) | Yes | ✅ Run, **our baseline** |
| B5 | MedGemma, asked in words | Show the photo to a medical AI chatbot and ask "how many cells?" | Yes (pre-trained) | No | Not run |
| B6 | MedSigLIP features + formula | A medical AI model describes the photo as numbers, and a simple formula turns those into a count | Partly (frozen pre-trained part plus a small learned part) | No | Not run |
| **B7** | YOLO26n | A general object detector, taught on our boxes | Yes (trained on our data) | Yes | Planned: our deep-learning model, trained on Colab next |

The ladder mixes two families of methods, which explains its order:
- **Count-only methods** (B0, B1, B2, B5, B6) output just numbers. They can't show *which* cells they counted, so nobody can check their work.
- **Methods that find cells** (B3, B4, B7) draw each cell. Their counts are tallies of what they found, which you can check against the picture.

**Why we ran B0, B1 and B4 first:**
- B0 and B1 cost nothing and are needed to interpret everything else.
- B4 is the "existing published model" the course asks for, so it is our **baseline**: every later model is compared against it.

B7 (YOLO) is the deep-learning approach we plan to build on; it is trained on Colab next. B2, B3, B5 and B6 are explained in §4b below.

---

## 4. The models one by one

### B0 · "Always guess the average"

**What it does:**
1. Look at the 351 training photos.
2. Compute the average number of cells per photo (about 54).
3. Predict exactly that number for **every** new photo, without looking at it.

**Analogy:** a weather forecast that always says "20 °C", because that's the yearly average.

**Why include something this silly?** It is the **floor**. Any real model must beat it clearly; if it doesn't, it hasn't learned anything useful.

**Result:** off by **12.7 cells** per photo on average, and R² ≈ 0, as expected for "always the average".

---

### B1 · "Guess the average of that session"

**What it does:**
1. For each of the 5 sessions, compute the average count in the training photos.
2. For a new photo, check which session it came from.
3. Predict that session's average.

It still never looks at the cells.

**Analogy:** a forecast that says "it's 30 °C in Phoenix and 10 °C in Seattle", using only the city's average, not today's sky.

**Why include it?** It's our **cheating detector**. During data exploration we found that the session alone says a lot about the count. If a fancy model scores close to B1, it may just be recognizing the session from the background colour instead of counting cells. A good model has to beat B1, and a model trained on our photos also has to hold up on the leave-one-session-out test.

**Result:** off by **6.9 cells**, R² **0.71**. Just knowing the session explains most of the variation, which shows how strong the shortcut is. On a session it has never seen, it would have no average to use, so it is useless there.

---

### B4 · Cellpose-SAM: a ready-made cell finder (our baseline model)

**What is Cellpose?** A free, published program from a university research group. Its authors trained a deep-learning model on many thousands of microscope images of all kinds of cells. Its job is to **outline every cell** in a photo; this is called *segmentation*. We downloaded it and used it **as is**, without teaching it anything about our photos.

> **"Pre-trained" / "off the shelf"** means someone else already did the expensive learning, and we just use the result. It's like buying a calculator instead of building one.

**What we did, step by step:**
1. **Download** the model: one file, about 1.2 GB, fetched automatically the first time.
2. **Run it on every photo.** It returns an outline for each object it thinks is a cell, taking about 2 seconds per photo.
3. **Pick two "strictness" settings on the training photos.** Cellpose has two settings that decide how sure it must be before it calls something a cell. With its normal settings it missed about two thirds of our cells, because many of them are pale and low-contrast. We tried 15 combinations on the **training photos** and kept the one that gave the most accurate counts there. It is much less strict than the default.
4. **Throw away tiny specks.** Outlines smaller than 400 pixels are dust or debris, not cells. The 400 was also chosen on the training photos.
5. **Count** the outlines that are left.

**What "learns from our data" here:** only those three numbers (two settings and the minimum size), all picked using the **training photos only**. The model itself is unchanged. The validation and test photos played no part.

**Why it is our baseline:** it answers **"Can we just use an existing published tool, with no training?"** That's the comparison the course asks for, and an important question for a lab that doesn't want to label hundreds of photos. Any model we build has to beat it.

**Result:**
- **Normal test:** off by **5.6 cells** per photo (about 11%), within ±10% on **59%** of photos, R² **0.74**. That's better than both B0 and B1, without ever being trained on our cells. On average it neither over- nor undercounts (bias −0.6).
- **With its normal settings:** off by **34.8 cells**. It finds only a third of the cells, so picking the settings matters a lot.
- **Weakness:** it misses pale, faint cells. On the worst photos it finds only about 2/3 of the cells.
- **Not a weakness here:** blurry photos. Its error barely changes with how sharp the photo is.
- **Speed:** about 1.85 seconds per photo, because it's a very large model.

**One thing that looks bad but isn't:** its precision and recall at IoU ≥ 0.5 are only about 0.65. That's because Cellpose draws its outline around each cell's faint halo, so its boxes are bigger than the hand-drawn ones (about 37 vs 29 pixels wide) and overlap them less. With a looser overlap rule (IoU ≥ 0.3), it scores about 0.9 for both. It finds most cells; it just draws them bigger.

> **Fixing a mistake:** in an earlier run we picked the two settings by eye on 5 photos, and 2 of those turned out to be test photos. That's like peeking at the final exam. We now pick them on the training photos only. The scores barely changed (5.6 → 5.6 cells; R² 0.76 → 0.74).

---

### B7 · YOLO26n: a general object detector, taught on our cells (planned deep-learning model)

**What is YOLO?** "You Only Look Once" is a popular family of **object detectors**: programs that look at a photo and draw a labelled box around each object they recognize. They're used in self-driving cars, phone cameras and security cameras. YOLO26 is the 2026 version from the company Ultralytics, and **"n" (nano)** is its smallest, fastest size.

**What does "deep learning" mean here, in one paragraph?**
- YOLO is a *neural network*: a very large mathematical function with millions of adjustable numbers, called *weights*.
- **Training** means showing it many photos together with the correct boxes, and nudging its weights a tiny bit each time so its boxes get closer to the correct ones.
- After many rounds it has "learned" what the objects look like. Nobody writes rules like "a cell is round"; the network works that out from the examples.

**Starting point: pre-trained on everyday photos.** We don't start from scratch. The model we download (5.5 MB) was already trained by Ultralytics on **COCO**, a public collection of about 120,000 everyday photos (people, cars, dogs, cups…). It has never seen a cell, but it already knows general visual skills: edges, round shapes, textures, and "an object versus the background". We then **teach it to find our cells**.

> **Analogy:** hiring someone who already knows how to read, then teaching them your company's forms. That's much faster than teaching someone to read from zero.
>
> This is called **transfer learning**, and the step of teaching the new task is often called **fine-tuning**. It means *adapting* the model to our cells, not polishing it for the best possible score.

**What we will do, step by step:**
1. **Convert our labels to YOLO's format.** For every photo, a small text file lists each box as "class, box centre, box width and height", using the same training, validation and test groups as everyone else.
2. **Train** on the 351 training photos for 100 rounds; one round through all the photos is an *epoch*. This takes roughly 15–30 minutes on a Colab GPU.
   - During training YOLO also creates varied copies of the photos, such as flipped, slightly recoloured or zoomed ones (*augmentation*), so it learns cells, not specific pixels.
   - After each round it checks itself on the validation photos and keeps the version that scored best.
3. **Keep every setting at the default.** No experimenting with settings: this is the plain, standard recipe, a starting point to improve on later.
4. **Choose the confidence cut-off.** For every box, YOLO also gives a **confidence** between 0 and 1: how sure it is that the box is a cell. We need a cut-off: "only count boxes above X".
   - Too low counts junk.
   - Too high misses real cells.
   - We try values on the **validation photos** and pick the one that makes the counts most accurate. The test photos are not used for this choice.
5. **Count:** the number of boxes above the cut-off.

**Why include it?** Deep learning on our own labels is the approach we plan to build on. YOLO learns from every one of the 26,715 hand-drawn boxes, and it is fast enough for real-time use. Starting from the standard, untuned recipe lets us measure each improvement we try next (more data variety, a bigger model, larger photos) one step at a time.

**Status:** not run in this round. It is trained on Colab next and will be scored on the same total-count metrics as Cellpose, plus the hidden-session test. It has to beat Cellpose's 5.6 cells on the normal test.

**Later extension:** YOLO can also label each box with a kind of cell (for example Live or Dead), which is how we would add viability later.

---

## 4b. The other rungs (planned, not run yet)

### B2 · "Measure the whole photo, then use a formula"

**What it would do:**
1. For each photo, compute a few overall measurements without finding any cells. For example:
   - **"Edge energy":** how much sharp light-to-dark change the photo has. Every cell has an outline, so more cells means more edges.
   - **Colour statistics:** average redness, greenness and blueness; how much the colours vary.
2. Fit a simple formula, a *linear regression* (a weighted sum of the measurements), on the 351 training photos: "count ≈ a × edge energy + b × blueness + …".
3. For a new photo, compute the same measurements and plug them into the formula.

**Analogy:** estimating how crowded a stadium is from how loud it sounds, without counting heads.

**Why it's on the ladder:** it shows how far you get **without ever locating a cell**. Our data exploration found that edge energy alone tracks the count very closely (correlation 0.90), so B2 might do surprisingly well on the normal test.

**What we expect:** decent on the normal test, but fragile. A blurrier photo has fewer sharp edges, so the formula would say "fewer cells" even when the count is the same. The background colour also gives away the session, so B2 could pick up the same shortcut as B1. It learns from only 351 numbers, one per photo.

**Why not run yet:** it isn't deep learning, which this week's check-in focuses on, and B1 already shows what a "don't look at cells" method can do. It's quick to add later if we want the full ladder.

---

### B3 · Classic image processing (hand-written rules, no learning)

**What it would do,** in the steps a person programs explicitly:
1. **Remove the background:** blur the photo heavily to get just the background tint, then subtract it, so cells stand out from an even background.
2. **Threshold:** turn the photo black-and-white, so every pixel is either "cell" or "background". A standard method called **Otsu's method** picks the cut-off automatically.
3. **Split touching cells:** a method called **watershed** cuts blobs that contain two cells stuck together.
4. **Count the blobs,** dropping specks that are too small or too large.
The only settings (blob size limits) would be chosen on the training photos. A "how blue is it?" colour rule could later add Live/Dead.

**Analogy:** giving someone a checklist ("dark round blob between 20 and 40 pixels = a cell") instead of letting them learn from examples.

**Why it's on the ladder:** this is how cell counting was automated before deep learning. Tools like ImageJ and CellProfiler work this way. It is the classic **non-deep-learning baseline** the course mentions. If B3 were nearly as good as Cellpose or YOLO, the extra complexity of deep learning wouldn't be worth it.

**What we expect:**
- It works on crisp, well-separated cells, which suits our data, since cells barely overlap.
- It breaks on pale live cells, which barely differ from the background, and on uneven lighting.
- It probably counts faint cells the labellers skipped, so it over-counts.
- Every new lab or microscope needs its settings re-adjusted by hand.

**Why not run yet:** Cellpose (B4) already covers the "no training on our labels" case with a stronger cell finder. B3 is the natural addition if the report wants a non-deep-learning comparison.

---

### B5 · MedGemma: ask a medical AI chatbot

**What is MedGemma?** A **vision-language model** from Google: an AI that can look at an image and answer questions about it in text, like ChatGPT with pictures. "Med" means it was further trained on medical images, such as X-rays, skin photos and pathology slides. The 4-billion-weight version can run on a strong laptop.

**What it would do:**
1. Give it a photo plus a question: *"How many cells are in this image? Answer with one number."*
2. Read the number out of its text answer.
3. No training at all; this is **prompting**, simply asking.

**Analogy:** asking a well-read medical student to glance at the slide and estimate, instead of counting carefully.

**Why it's on the ladder:** it answers the question stakeholders will ask: **"Why build anything? Can't a big AI model just do it?"** Having the real number makes that answer convincing.

**What we expect: weak.**
- Models like this are known to be unreliable at counting more than about 10 objects; our photos have 23–91.
- It was trained on different kinds of medical images, not trypan-blue cell cultures.
- Its answers can change with the wording of the question.
- It's also much slower and heavier than Cellpose or YOLO.

**Why not run yet:** it needs a Hugging Face account and acceptance of Google's terms of use before download. It's a stretch goal in the plan (§5.5).

---

### B6 · MedSigLIP features + a small formula

**What is MedSigLIP?** The "eyes" part of MedGemma: an image model that turns any picture into a long list of numbers, called an **embedding** or **features**, summarizing what's in it. Similar pictures get similar lists.

**What it would do:**
1. Pass every photo through MedSigLIP **frozen** (unchanged; we don't train it) to get its list of numbers.
2. On the 351 training photos, fit a small model that maps those numbers to a cell count. The small model is a regression formula (*ridge regression*) or a tiny neural network (*MLP*); only this small part learns.
3. For a new photo: get its numbers from MedSigLIP, then apply the small model.

**Analogy:** an art expert writes a detailed description of each photo, and you learn which words in the description go with high or low counts.

**Why it's on the ladder:** it's the cheapest way to use a big pre-trained medical model *with* our data: training takes minutes, not hours. It shows whether general "medical image understanding" helps with counting.

**What we expect:**
- Probably good on the normal test, but at high risk of the **session shortcut**. A photo's overall description strongly reflects its background colour, so the formula could learn "this tint means about 70 cells", just like B1.
- The hidden-session test is the real judge.
- Like all count-only methods, it can't show which cells it counted, and it learns from just 351 numbers (one per photo), versus 26,715 boxes for YOLO.

**Why not run yet:** the same Hugging Face access step as B5. It's part of the same stretch track (plan §5.5).

---

### Why "finds individual cells" matters so much for our data

| | Count-only methods (B0, B1, B2, B5, B6) | Methods that find cells (B3, B4, B7) |
|---|---|---|
| **Learns from** | 1 number per photo, so **351 examples** | Every cell, so **26,715 examples** (for B7; Cellpose learned from other labs' cells) |
| **Can it cheat with the session?** | Easily: the overall look of the photo gives the session away | Harder: each cell has to be found |
| **Can a person check its answer?** | No, it only outputs a number | Yes: look at the boxes on the photo |
| **Works on a bigger or cropped photo?** | Not reliably: trained on one photo size | Yes: it just finds more or fewer cells |

This is why both our baseline (B4, Cellpose) and our planned deep-learning model (B7, YOLO) find cells and then count them. The count-only rungs are there to show that choice was justified, not as serious candidates.

---

## 5. Putting it together

| | Normal test: off by (cells) | Within ±10% | R² | Time per photo |
|---|---|---|---|---|
| B0 average | 12.7 | 24% | −0.01 | — |
| B1 session average | 6.9 | 45% | 0.71 | — |
| Cellpose with its normal settings | 34.8 | 9% | −6.21 | 1.85 s |
| **B4 Cellpose (baseline; settings picked on training photos)** | **5.6** | **59%** | **0.74** | **1.85 s** |
| B7 YOLO (trained on our cells) | next, on Colab | | | |

**The story the table tells:**
1. **Guessing is not enough (B0).** Counts vary too much from photo to photo.
2. **The session shortcut is strong but fake (B1).** It looks decent on the normal test without looking at a single cell, and it can't work on a new session. Any model has to beat it.
3. **A ready-made tool gets you a good part of the way (B4)** with no labelling effort, once two settings are adapted to our photos. It misses faint cells and is slow.
4. **Next: teach a detector on our own labels (B7).** The question is how much it beats Cellpose, especially on unseen sessions and on messier photos (blurry, differently stained, different microscopes).

---

## 6. How to read the figures in `baseline_results.md`

- **Threshold grid (heat map):** each square is one combination of Cellpose's two settings (how cell-like a pixel must be, and how well-formed an outline must be), coloured by how far off the count was on the training photos. Darker is better. The top-left square is Cellpose's normal settings.
- **Predicted vs true (scatter plots):**
  - Each dot is one test photo, placed at (true count, predicted count).
  - A perfect model puts every dot on the dashed diagonal line.
  - B1 makes 5 flat steps, one per session. Cellpose with normal settings sits far below the line. Tuned Cellpose scatters around it.
- **Error by session / count / sharpness (bar charts):** a tall bar means a big error for that group of photos. They show whether a model fails on particular sessions, on crowded photos, or on blurry photos.
- **Error vs sharpness and count (scatter plots):** each dot is a test photo; above the zero line means overcounted, below means undercounted.
- **Effect of the two settings (four photos side by side):** the same photo under Cellpose's normal settings, with each setting loosened alone, and with both loosened. Only with both loosened does it find nearly all the cells.
- **Worst photos (image grids):** one grid for Cellpose with its normal settings (it finds almost only the dark-blue cells), one for the tuned baseline.
  - **Yellow boxes** are the human labels.
  - **Cyan boxes** are the cells Cellpose found.
  - A yellow box with no cyan box means a **missed cell**. A cyan box with no yellow box means an **extra** detection.

---

## 7. Mini glossary

| Term | Meaning |
|---|---|
| **Model** | A program that takes a photo and produces an answer (here: cell boxes and counts) |
| **Label / ground truth** | The correct answer, drawn by a person |
| **Training** | Letting the model adjust itself using photos with known answers |
| **Pre-trained** | Already trained by someone else on other data; we start from their result |
| **Fine-tuning / transfer learning** | Continuing to train a pre-trained model on our own data so it learns our task |
| **Zero-shot / off the shelf** | Using a pre-trained model as is, without training it on our data |
| **Epoch** | One full pass of training over all the training photos |
| **Augmentation** | Making varied copies of training photos (flipped, recoloured, zoomed) so the model generalizes |
| **Overfitting** | Memorizing the training photos instead of learning the general pattern; looks great on homework, fails the exam |
| **Confidence** | The model's own 0–1 score for how sure it is about a box |
| **Segmentation** | Outlining the exact shape of each object (what Cellpose does) |
| **Object detection** | Drawing a labelled box around each object (what YOLO does) |
| **GPU** | A graphics chip that runs deep-learning maths much faster than a normal processor |
| **Baseline** | A simple or standard method we compare later improvements against |
