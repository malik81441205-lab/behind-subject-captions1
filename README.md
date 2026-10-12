# Behind-Subject Captions

Puts caption text **behind the person** in a video (depth / 3D look). Templates are JSON files, so adding a style never touches the engine.

## How it works
Per frame: `background -> caption text (behind) -> person (cut out with a mask) -> caption text (front)`.
The person is the original frame shown only where the segmentation mask is 1, so they cover the text.

## Install
```
pip install -r requirements.txt
# ffmpeg must be on PATH (apt install ffmpeg / brew install ffmpeg)
```
Fonts (Anton, Montserrat, Playfair Display, all OFL/free) are in `fonts/`.

## Run (automatic captions - no .srt needed)
```
python main.py --video talk.mp4 --template dynamic --out dynamic.mp4
```
It listens to the whole video (Whisper, runs locally, free), finds the exact time of every word, and places captions to match the voice. Captions break at sentence ends and pauses; the keyword is the longest meaningful word (filler like "the / we / are" is skipped).

Useful options:
```
--model small        # tiny | base (default) | small | medium : bigger = more accurate, slower
--language en        # force a language (default: auto-detect), e.g. ur, hi, ar, es
--preview 2.5        # render only one frame (preview.png) to check the look fast
--transcribe-only    # just save the transcript + talk.srt, no video
--srt my.srt         # use your own caption file instead of auto-transcribing
--set '{"colors":{"keyword":"#FF3366"}}'   # custom settings over any template
python main.py --list
```
The transcript is saved next to the video as `talk.words.json`, so re-rendering with another template is instant. **To fix a misheard word, open that file, edit the text, and render again.** Delete the file to transcribe again.

## Templates (12)
Run `python main.py --list`. Preview them all: `python tools/preview_all.py video.mp4 captions.srt 1.4 mediapipe` -> `template_sheet.png`.

| name | look |
|---|---|
| `dynamic` | Brief template 1: condensed caps, filled orange keyword pill, word pop, slight tilt |
| `luxe` | Brief template 2: tracked small caps + big teal-gradient keyword, smooth fade/slide |
| `hind` | Soft white text + glowing cyan italic keyword |
| `boxed` | Lime serif-italic keyword inside a guide-line frame with corner dots |
| `sliding_highlight` | Yellow highlight bar slides to each word as it's spoken |
| `master` | Small lead-in + big cream-to-blue gradient keyword |
| `simple` | Dark rounded card, one big word + small blue accent |
| `podcast` | Left-aligned stacked heavy caps, upcoming words dimmed, lit as spoken |
| `karaoke_pop` | Outlined caps, spoken word flashes yellow and punches up |
| `neon_noir` | Magenta/cyan neon glow, tracked caps, spins in |
| `gold_estate` | Luxury serif keyword with gold gradient |
| `comic_pow` | Sticker style: fat yellow caps, hard shadow, bouncing drop-in |

### Custom settings (what a website "Edit" panel would change)
Any value in a template can be overridden per render, without touching the template file:
```
python main.py --video v.mp4 --srt v.srt --template hind \
  --set '{"colors":{"keyword":"#FF3366"},"layout":{"y_ratio":0.4},"behind":"all"}'
```
In code: `load_template(name, override_dict)`. Your editor UI maps its controls onto these keys.

### Template reference
| key | meaning |
|---|---|
| `font`, `uppercase`, `colors.text/keyword` | base look |
| `styles.<role>` / `keyword` | a text role: `font`, `scale`, `color`, `gradient:[top,bottom]`, `pill`, `own_line`, `uppercase`, `tracking`, `effects` |
| `pattern` + `keyword_role:false` | cycle roles by word position (e.g. big / small / base) |
| `behind` | `smart` (keyword + the words below it go behind the person, so nothing covers the face), `keyword`, `all` or `none` |
| `layout.auto_place` | default on: the caption is placed just above the person's head (the head covers only the bottom ~35% of the keyword, so it stays readable). Tune with `{"auto_place":{"overlap":0.5}}` or turn off with `false` to use `y_ratio` |
| `active` | style of the word being spoken (`color`, `scale`) |
| `upcoming` | `{"mode":"dim","color":..}` shows all words from the start, dimmed until spoken |
| `highlight` | sliding rounded bar behind the spoken word |
| `box` / `frame` | card behind the whole caption / guide-line frame with corner dots |
| `tilt` | degrees; alternates sign each caption |
| `effects.stroke/glow/shadow` | all sizes are in "em" (fractions of the font size), so they scale with the video |
| `size.font_height_ratio`, `max_width_ratio`, `max_height_ratio` | auto-size: starts at a fraction of video height, shrinks until it fits width AND height |
| `layout.y_ratio / x_ratio / align / max_words_per_line` | position (used when `auto_place` is off or nobody is in frame) |
| `animation.type` | `pop`, `fade`, `fade_slide`, `slide_left`, `zoom_out`, `drop`, `spin_in`, `blur_rise`, `none` (+ `duration`, `easing`, `stagger: word/chunk`) |

Templates are generated from `tools/make_templates.py` for convenience, but the JSON files in `templates/` are the source of truth - edit them directly or copy one to make a new style.
New animation = one function in `renderer/animations.py` + one line in `ANIMATIONS`.

## Captions / keywords
Auto mode uses real word timestamps. If you pass an SRT instead, word timing is spread across each cue by length. The keyword is the longest word in each chunk, or mark it in the SRT: `Discover the *power* of AI`.

## Swapping the segmentation model
Implement `Segmenter._predict(frame_bgr) -> float mask [0,1]` in `segmentation/`, register in `segmentation/__init__.py`.
MediaPipe = fast CPU. `rembg` = better edges, slower. `rvm` = Robust Video Matting, best quality, use a GPU (needs torch).

## Structure
```
main.py  captions.py  transcribe.py  compositor.py  template_loader.py
segmentation/ (base, mediapipe, rembg, rvm, debug)
renderer/ (text_layer = layout + effects + autosize, animations)
templates/*.json   fonts/   samples/   tools/ (preview_all.py, make_templates.py)
```

## Known limits / next steps
- Frames with no active caption skip segmentation (faster). Speed on long videos: use a GPU model, or segment at lower res and upscale the mask.
- Best with a single person, good contrast and lighting.
- Not yet tested on real footage in this repo: run it on a real talking-head clip and tune `edge_blur` / `temporal_smoothing` in `segmentation/base.py`.
