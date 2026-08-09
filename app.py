"""
Sign2Text — full app.

A single-file Gradio app with:
  - a login gate (any non-empty username/password passes)
  - a sidebar-navigated multi-page layout (Home, Recognition, How It Works,
    Model, Performance, Future Scope, About)
  - a dark/light theme toggle
  - the live webcam sign-recognition tool from the earlier version

Everything is pure Python. Theme switching and page navigation use Gradio's
own "return an updated component" mechanism (e.g. `return gr.Column(visible=False)`)
instead of any hand-written JavaScript.

Run with:
    python app.py
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import os
import sys

# Windows' default console codepage (cp1252) can't print the emoji used in
# this app's labels, which crashes when Python tries to print anything
# containing them (including its own error messages). Force UTF-8 output
# so that doesn't happen, regardless of the terminal's configured codepage.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import cv2
import gradio as gr
import mediapipe as mp
import numpy as np

try:
    from tensorflow.keras.models import load_model
except ImportError:  # pragma: no cover
    load_model = None

# --------------------------------------------------------------------------
# Model loading (unchanged from the original recognition script)
# --------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_language_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "gesture_labels.npy")

CONFIDENCE_THRESHOLD = 0.7
STABLE_FRAMES_REQUIRED = 4

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

model = None
gesture_labels = None

if load_model is not None and os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    model = load_model(MODEL_PATH)
    gesture_labels = np.load(LABELS_PATH, allow_pickle=True).item()
    print(f"[startup] Model loaded with gestures: {list(gesture_labels.values())}")
else:
    print("[startup] WARNING: sign_language_model.h5 / gesture_labels.npy not found. "
          "The Recognition page will run hand tracking but won't show predictions "
          "until you add both files next to app.py.")


def process_frame(frame, transcript, last_gesture, stable_count, last_committed):
    if frame is None:
        return None, "—", 0.0, transcript, transcript, last_gesture, stable_count, last_committed

    results = hands.process(frame)
    annotated = frame.copy()
    gesture_text = "—"
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            if model is not None:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_arr = np.array(landmarks).reshape(1, -1)

                prediction = model.predict(landmarks_arr, verbose=0)
                gesture_idx = int(np.argmax(prediction))
                pred_confidence = float(prediction[0][gesture_idx])

                if pred_confidence > CONFIDENCE_THRESHOLD:
                    gesture_text = str(gesture_labels[gesture_idx])
                    confidence = pred_confidence

        if gesture_text == last_gesture:
            stable_count += 1
        else:
            last_gesture = gesture_text
            stable_count = 1

        if (
            gesture_text != "—"
            and stable_count == STABLE_FRAMES_REQUIRED
            and gesture_text != last_committed
        ):
            transcript += gesture_text
            last_committed = gesture_text
    else:
        last_gesture = None
        stable_count = 0

    return annotated, gesture_text, round(confidence, 3), transcript, transcript, last_gesture, stable_count, last_committed


def clear_transcript():
    return "", None, 0, None


# --------------------------------------------------------------------------
# Styling — Sign2Text palette, both theme variants
# --------------------------------------------------------------------------

CUSTOM_CSS = """
#app-shell.theme-dark {
    --bg-canvas: #08191E;
    --bg-panel: #0B2027;
    --bg-panel-alt: #0F2932;
    --text-primary: #FFFFFF;
    --text-secondary: #B0C4DE;
    --accent-gold: #E2B15B;
    --accent-amber: #F5C467;
    --cta-glow: #38A3A5;
    --cta-border: #26828E;
    --nav-active: #8A6632;
    --status-live: #2A9D8F;
    --border-subtle: rgba(226, 177, 91, 0.18);
}

#app-shell.theme-light {
    --bg-canvas: #F6F3EC;
    --bg-panel: #FFFFFF;
    --bg-panel-alt: #FBF6EC;
    --text-primary: #14232A;
    --text-secondary: #4B5A63;
    --accent-gold: #B4791E;
    --accent-amber: #C98B2A;
    --cta-glow: #38A3A5;
    --cta-border: #26828E;
    --nav-active: #E9D2A7;
    --status-live: #2A9D8F;
    --border-subtle: rgba(180, 121, 30, 0.25);
}

#app-shell {
    background: var(--bg-canvas);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
    border-radius: 14px;
    padding: 0 !important;
}

.s2t-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 22px;
    border-bottom: 1px solid var(--border-subtle);
}

.s2t-brand {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: var(--accent-gold);
}

.s2t-brand span {
    color: var(--text-primary);
    font-weight: 400;
    font-size: 0.85rem;
    margin-left: 8px;
}

.s2t-sidebar {
    background: var(--bg-panel);
    border-right: 1px solid var(--border-subtle);
    padding: 16px 10px !important;
    min-height: 560px;
    overflow: hidden;
    transition: min-width 0.3s ease, width 0.3s ease, padding 0.3s ease;
}

.s2t-sidebar-hidden {
    min-width: 0 !important;
    width: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    border-right: none !important;
}

.s2t-menu-toggle button {
    background: transparent !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
    font-size: 1rem !important;
}

.s2t-sidebar button {
    text-align: left !important;
    justify-content: flex-start !important;
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 0.92rem !important;
    padding: 10px 12px !important;
    margin-bottom: 2px !important;
    border-radius: 8px !important;
}

.s2t-sidebar button:hover {
    background: var(--border-subtle) !important;
    color: var(--accent-amber) !important;
}

.s2t-nav-active button {
    background: var(--nav-active) !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

.s2t-content {
    padding: 26px 30px !important;
    background: var(--bg-canvas);
}

.s2t-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 18px 20px !important;
}

.s2t-cta-primary button {
    background: var(--bg-panel) !important;
    color: var(--accent-amber) !important;
    border: 1.5px solid var(--cta-border) !important;
    box-shadow: 0 0 14px var(--cta-glow), 0 0 2px var(--cta-glow) inset !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 16px !important;
}

.s2t-cta-primary button:hover {
    box-shadow: 0 0 22px var(--cta-glow), 0 0 4px var(--cta-glow) inset !important;
}

.s2t-theme-toggle button, .s2t-logout button {
    background: transparent !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
}

.s2t-theme-pill {
    display: inline-flex !important;
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 999px;
    padding: 3px;
    gap: 3px;
    width: fit-content !important;
    flex-wrap: nowrap !important;
}

.s2t-theme-pill button {
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    border-radius: 50% !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    font-size: 0.95rem !important;
    line-height: 1 !important;
}

.s2t-theme-pill .s2t-toggle-active {
    background: var(--bg-canvas) !important;
    box-shadow: 0 0 10px var(--accent-gold) !important;
}

.s2t-hero {
    text-align: center;
    padding: 20px 10px 6px;
}

.s2t-hero-icon {
    font-size: 2.3rem;
    color: var(--cta-glow);
    text-shadow: 0 0 18px var(--cta-glow);
    line-height: 1;
    margin-bottom: 6px;
}

.s2t-hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: 0.01em;
    color: var(--accent-amber);
    text-shadow: 0 0 22px var(--accent-gold), 0 0 6px var(--accent-gold);
    margin: 0 0 14px;
}

.s2t-hero-sub {
    color: var(--text-secondary);
    font-size: 0.95rem;
    max-width: 620px;
    margin: 0 auto;
    line-height: 1.65;
}

.s2t-feature-row {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    justify-content: center;
    margin: 28px 0 6px;
}

.s2t-feature-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 18px 22px;
    min-width: 150px;
    text-align: center;
}

.s2t-feature-icon {
    font-size: 1.6rem;
    margin-bottom: 8px;
}

.s2t-feature-label {
    color: var(--accent-gold);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    font-weight: 700;
    margin-bottom: 4px;
}

.s2t-feature-value {
    color: var(--text-secondary);
    font-size: 0.85rem;
}

.s2t-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: monospace;
    font-size: 0.78rem;
    color: var(--status-live);
    letter-spacing: 0.06em;
}

.s2t-status-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--status-live);
    animation: s2t-pulse 1.6s ease-in-out infinite;
}

@keyframes s2t-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(42, 157, 143, 0.6); }
    70%  { box-shadow: 0 0 0 8px rgba(42, 157, 143, 0); }
    100% { box-shadow: 0 0 0 0 rgba(42, 157, 143, 0); }
}

#login-screen {
    max-width: 420px;
    margin: 60px auto;
    background: #0B2027;
    border: 1px solid rgba(226, 177, 91, 0.25);
    border-radius: 16px;
    padding: 36px 34px !important;
}

#login-screen h1 {
    color: #E2B15B;
}

.s2t-chip {
    display: inline-block;
    font-size: 0.78rem;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid var(--border-subtle);
    color: var(--accent-amber);
    margin: 2px 4px 2px 0;
}
"""

PAGES = ["home", "recognition", "how_it_works", "model", "performance", "future_scope", "about"]
NAV_LABELS = {
    "home": "🏠 Home",
    "recognition": "🤟 Recognition",
    "how_it_works": "📊 How It Works",
    "model": "🧠 Model",
    "performance": "📈 Performance",
    "future_scope": "🚀 Future Scope",
    "about": "👩‍💻 About",
}


def navigate(page):
    """Return visibility updates for every page column, plus the sidebar
    button styling so the active page is highlighted."""
    visibility_updates = [gr.update(visible=(p == page)) for p in PAGES]
    nav_updates = [
        gr.update(elem_classes=["s2t-nav-active"] if p == page else [])
        for p in PAGES
    ]
    return visibility_updates + nav_updates


def set_theme(mode):
    return (
        mode,
        gr.update(elem_classes=[f"theme-{mode}"]),
        gr.update(elem_classes=["s2t-toggle-active"] if mode == "light" else []),
        gr.update(elem_classes=["s2t-toggle-active"] if mode == "dark" else []),
    )


def do_login(username, password):
    if username.strip() and password.strip():
        return gr.update(visible=False), gr.update(visible=True), ""
    return gr.update(visible=True), gr.update(visible=False), "⚠️ Enter both a username and password to continue."


def do_logout():
    return gr.update(visible=True), gr.update(visible=False)


def toggle_sidebar(is_open):
    is_open = not is_open
    classes = ["s2t-sidebar"] if is_open else ["s2t-sidebar", "s2t-sidebar-hidden"]
    return is_open, gr.update(elem_classes=classes)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

with gr.Blocks(title="Sign2Text", css=CUSTOM_CSS) as demo:

    theme_state = gr.State("dark")
    sidebar_state = gr.State(True)

    # ---------------- Login screen ----------------
    with gr.Column(visible=True, elem_id="login-screen") as login_screen:
        gr.Markdown("# Sign2Text")
        gr.Markdown("Sign in to continue. (Demo login — any username & password works.)")
        username_box = gr.Textbox(label="Username")
        password_box = gr.Textbox(label="Password", type="password")
        login_error = gr.Markdown("")
        login_btn = gr.Button("Log in", elem_classes=["s2t-cta-primary"])

    # ---------------- Main app shell ----------------
    with gr.Column(visible=False, elem_id="app-shell", elem_classes=["theme-dark"]) as app_shell:

        with gr.Row(elem_classes=["s2t-topbar"]):
            menu_btn = gr.Button("☰", elem_classes=["s2t-menu-toggle"], size="sm")
            gr.Markdown(
                "<div class='s2t-brand'>Sign2Text "
                "<span>AI-powered hand gesture &amp; sign language recognition</span></div>"
            )
            with gr.Row():
                with gr.Row(elem_classes=["s2t-theme-pill"]):
                    sun_btn = gr.Button("☀️", elem_classes=[], size="sm")
                    moon_btn = gr.Button("🌙", elem_classes=["s2t-toggle-active"], size="sm")
                logout_btn = gr.Button("Log out", elem_classes=["s2t-logout"], size="sm")

        with gr.Row():
            # ---------------- Sidebar ----------------
            with gr.Column(scale=0, min_width=220, elem_classes=["s2t-sidebar"]) as sidebar:
                nav_buttons = {}
                for page in PAGES:
                    nav_buttons[page] = gr.Button(NAV_LABELS[page])

            # ---------------- Content area ----------------
            with gr.Column(scale=4, elem_classes=["s2t-content"]):

                page_columns = {}

                # ---- Home ----
                with gr.Column(visible=True) as page_columns["home"]:
                    gesture_count = len(gesture_labels) if gesture_labels else 0
                    gr.Markdown(
                        "<div class='s2t-hero'>"
                        "<div class='s2t-hero-icon'>⬡</div>"
                        "<h1 class='s2t-hero-title'>Sign Language Recognition</h1>"
                        "<p class='s2t-hero-sub'>An AI-powered system that recognizes hand "
                        "gestures and converts sign language into readable text.<br>"
                        "It uses computer vision and machine learning to enable faster and "
                        "more accessible communication.</p>"
                        "</div>"
                    )

                    feature_cards = [
                        ("🧠", "MODEL", "Keras Dense NN"),
                        ("✋", "TRACKING", "MediaPipe Hands"),
                        ("📷", "CAMERA", "Live Webcam"),
                        ("⚡", "DETECTION", "Real-Time"),
                        ("🤟", "GESTURES", f"{gesture_count} Signs" if gesture_count else "—"),
                    ]
                    cards_html = "<div class='s2t-feature-row'>" + "".join(
                        f"<div class='s2t-feature-card'>"
                        f"<div class='s2t-feature-icon'>{icon}</div>"
                        f"<div class='s2t-feature-label'>{label}</div>"
                        f"<div class='s2t-feature-value'>{value}</div>"
                        f"</div>"
                        for icon, label, value in feature_cards
                    ) + "</div>"
                    gr.Markdown(cards_html)

                    gr.Markdown("### Main objective")
                    gr.Markdown(
                        "Recognize hand gestures in real time from a webcam feed and "
                        "translate them into text, making sign language communication "
                        "more accessible without specialized hardware."
                    )
                    start_btn = gr.Button(
                        "🚀 Get Started — Start Recognition",
                        elem_classes=["s2t-cta-primary"],
                    )

                # ---- Recognition ----
                with gr.Column(visible=False) as page_columns["recognition"]:
                    gr.Markdown("## 🤟 Live Recognition")
                    gr.Markdown(
                        "<div class='s2t-status-badge'><span class='s2t-status-dot'></span>"
                        "STREAMING</div>"
                    )
                    if gesture_labels:
                        gr.Markdown(
                            "**Trained gestures:** " + ", ".join(str(g) for g in gesture_labels.values())
                        )
                    else:
                        gr.Markdown(
                            "⚠️ No model found — add `sign_language_model.h5` and "
                            "`gesture_labels.npy` next to `app.py` to enable predictions."
                        )

                    transcript_state = gr.State("")
                    last_gesture_state = gr.State(None)
                    stable_count_state = gr.State(0)
                    last_committed_state = gr.State(None)

                    with gr.Row():
                        with gr.Column(elem_classes=["s2t-panel"]):
                            webcam_input = gr.Image(
                                sources=["webcam"], streaming=True, label="Camera", type="numpy"
                            )
                        with gr.Column(elem_classes=["s2t-panel"]):
                            annotated_output = gr.Image(label="Tracked hand", type="numpy")
                            with gr.Row():
                                gesture_output = gr.Textbox(label="Predicted sign", value="—", interactive=False)
                                confidence_output = gr.Number(label="Confidence score", value=0.0, interactive=False)

                    with gr.Column(elem_classes=["s2t-panel"]):
                        transcript_output = gr.Textbox(label="Transcript", value="", interactive=False, lines=2)
                        clear_btn = gr.Button("Clear transcript", size="sm")

                    webcam_input.stream(
                        fn=process_frame,
                        inputs=[webcam_input, transcript_state, last_gesture_state, stable_count_state, last_committed_state],
                        outputs=[annotated_output, gesture_output, confidence_output, transcript_output, transcript_state, last_gesture_state, stable_count_state, last_committed_state],
                    )
                    clear_btn.click(
                        fn=clear_transcript,
                        outputs=[transcript_output, last_gesture_state, stable_count_state, last_committed_state],
                    )

                # ---- How It Works ----
                with gr.Column(visible=False) as page_columns["how_it_works"]:
                    gr.Markdown("## 📊 How It Works")
                    gr.Markdown(
                        "**1. Hand detection** — MediaPipe locates the hand in each webcam "
                        "frame.\n\n"
                        "**2. Landmark extraction** — 21 (x, y, z) landmark points are pulled "
                        "from the detected hand, giving a 63-number representation of its "
                        "shape and position.\n\n"
                        "**3. ML model prediction** — those 63 numbers are fed into a trained "
                        "neural network, which outputs a probability for each trained "
                        "gesture.\n\n"
                        "**4. Text conversion** — the highest-probability gesture (above a "
                        "confidence threshold) is shown as text and appended to the running "
                        "transcript."
                    )

                # ---- Model ----
                with gr.Column(visible=False) as page_columns["model"]:
                    gr.Markdown("## 🧠 Model")
                    gr.Markdown(
                        "**Dataset:** _[Fill in: number of samples per gesture, how many "
                        "gestures, how the data was collected]_\n\n"
                        "**Technologies used:** Python, OpenCV, MediaPipe, TensorFlow/Keras, Gradio\n\n"
                        "**Architecture:** Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → "
                        "Dropout(0.3) → Dense(softmax) — a feed-forward network over the "
                        "63 hand-landmark coordinates.\n\n"
                        "**Accuracy:** _[Fill in your validation accuracy from training]_"
                    )

                # ---- Performance ----
                with gr.Column(visible=False) as page_columns["performance"]:
                    gr.Markdown("## 📈 Performance")
                    gr.Markdown(
                        "_Add your real evaluation results here — this section is a "
                        "placeholder since these numbers depend on your trained model._"
                    )
                    gr.Markdown(
                        "- **Accuracy:** _[value]_\n"
                        "- **Precision:** _[value]_\n"
                        "- **Recall:** _[value]_\n"
                        "- **Confusion matrix:** _[insert image or table]_\n"
                        "- **Prediction examples:** _[insert sample screenshots or a table "
                        "of gesture → predicted label]_"
                    )

                # ---- Future Scope ----
                with gr.Column(visible=False) as page_columns["future_scope"]:
                    gr.Markdown("## 🚀 Future Scope")
                    gr.Markdown(
                        "- **Sentence-level recognition** — recognize sequences of gestures "
                        "as full sentences, not just isolated letters/words.\n"
                        "- **Voice output** — read the transcript aloud with text-to-speech.\n"
                        "- **More signs/gestures** — expand beyond the current label set.\n"
                        "- **Multilingual translation** — translate recognized text into "
                        "other spoken/written languages.\n"
                        "- **Mobile deployment** — package the model for on-device use "
                        "(e.g. TensorFlow Lite) on phones."
                    )

                # ---- About ----
                with gr.Column(visible=False) as page_columns["about"]:
                    gr.Markdown("## 👩‍💻 About")
                    gr.Markdown(
                        "**Developer:** _[Your name]_\n\n"
                        "**Project motivation:** _[Why you built this]_\n\n"
                        "**GitHub / contact:** _[link]_"
                    )

        menu_btn.click(
            fn=toggle_sidebar,
            inputs=[sidebar_state],
            outputs=[sidebar_state, sidebar],
        )

        # Wire sidebar nav buttons
        page_order = list(page_columns.keys())
        for page in page_order:
            nav_buttons[page].click(
                fn=lambda p=page: navigate(p),
                outputs=[page_columns[p] for p in page_order] + [nav_buttons[p] for p in page_order],
            )

        # Home's "Get Started" button jumps straight to Recognition
        start_btn.click(
            fn=lambda: navigate("recognition"),
            outputs=[page_columns[p] for p in page_order] + [nav_buttons[p] for p in page_order],
        )

        sun_btn.click(
            fn=lambda: set_theme("light"),
            outputs=[theme_state, app_shell, sun_btn, moon_btn],
        )
        moon_btn.click(
            fn=lambda: set_theme("dark"),
            outputs=[theme_state, app_shell, sun_btn, moon_btn],
        )

    login_btn.click(
        fn=do_login,
        inputs=[username_box, password_box],
        outputs=[login_screen, app_shell, login_error],
    )
    logout_btn.click(fn=do_logout, outputs=[login_screen, app_shell])

if __name__ == "__main__":
    demo.launch()