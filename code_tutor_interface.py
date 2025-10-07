import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" # internvl-3-14b needs two 80g A100/H100 GPUs
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tempfile
import base64
import re
import streamlit as st
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from lmdeploy.vl import load_image

# =================== basic setting ===================
MODEL = 'OpenGVLab/InternVL3-14B'
IMAGE_PATH = 'xxx'
PROBLEM = (
    "Problem: Compute the required sum using a loop. "
    "Student uses block code in the image."
)

# generate parameters: use GenerationConfig
ASSESS_GEN = GenerationConfig(max_new_tokens=600, temperature=0.2, top_p=0.9)
HINTS_FALLBACK_GEN = GenerationConfig(max_new_tokens=200, temperature=0.2, top_p=0.9)
QA_GEN = GenerationConfig(max_new_tokens=400, temperature=0.2, top_p=0.9)

# =================== loading model ===================
@st.cache_resource(show_spinner="Loading VLM model… (first time only, may take a while)")
def load_model():
    return pipeline(
        MODEL,
        backend_config=TurbomindEngineConfig(session_len=4096, tp=2),
        chat_template_config=ChatTemplateConfig(model_name='internvl3')
    )

pipe = load_model()

ASSESS_PROMPT = """You are a coding coach for scratch block-based programming. For example, "change number by 1" means increment the number by 1.
You will receive:
- The problem statement
- Conversation history so far
- The student's current code (from screenshot initially, then as text revisions)

Task:
1) Read the student's code and convert it to text code for better understanding.
2) Assess whether the student's code is correct or not.
3) If there are problems with the code, produce 3 hints from vague -> specific. Keep each hint <= 2 sentences.

Return in plain text with sections:
[ASSESSMENT]
...
[HINTS]
1) ...
2) ...
3) ...
"""

QA_PROMPT = """You are a coding coach for block-based programming. 
Always provide a short, clear, and helpful answer to the student's question. 
Do not repeat the question. Do not output section markers. 
Start your answer directly under [COACH ANSWER]."""

def _extract_hints(text: str):
    if not text:
        return []
    hints = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r'^\s*(?:[-*•]\s*)?(?:\(?\s*(\d+)\s*[\)\.]?)\s*(.+)$', s)
        if m:
            num, content = m.group(1), m.group(2).strip()
            if num in {"1", "2", "3"} and content:
                hints.append(content)
        elif re.match(r'^\s*hint\s*[:\-]\s*(.+)$', s, flags=re.IGNORECASE):
            hints.append(re.sub(r'^\s*hint\s*[:\-]\s*', '', s, flags=re.IGNORECASE).strip())
        if len(hints) >= 3:
            break
    return hints[:3]

def _truncate_history(history_text, max_chars=3000):
    return history_text if len(history_text) <= max_chars else history_text[-max_chars:]

def assess_and_hints(problem_text, img, history_text="", revised_code_text=None):
    history_text = _truncate_history(history_text)
    if revised_code_text is not None:
        body = (
            f"{problem_text}\n\n"
            f"[CONVERSATION HISTORY]\n{history_text}\n\n"
            f"[CURRENT STUDENT CODE - TEXT]\n{revised_code_text}\n\n"
            f"{ASSESS_PROMPT}"
        )
        out = pipe((body, img), gen_config=ASSESS_GEN).text
    else:
        body = (
            f"{problem_text}\n\n"
            f"[CONVERSATION HISTORY]\n{history_text}\n\n"
            f"[CURRENT STUDENT CODE - IMAGE]\nThe student's current code is shown in the image.\n\n"
            f"{ASSESS_PROMPT}"
        )
        out = pipe((body, img), gen_config=ASSESS_GEN).text

    hints = _extract_hints(out)
    if not hints:
        fb_prompt = (
            "From the assessment above, list exactly 3 brief hints for the student, "
            "from vague to specific, each 1 sentence. Use the format:\n"
            "[HINTS]\n1) ...\n2) ...\n3) ...\n"
        )
        fb_out = pipe((fb_prompt + "\n\n" + out, img), gen_config=HINTS_FALLBACK_GEN).text
        hints = _extract_hints(fb_out) or []
    return out, hints[:3]

def answer_question(question, history_text="", img=None):
    history_text = _truncate_history(history_text)
    prompt = (
        f"{QA_PROMPT}\n\n"
        f"[CONVERSATION HISTORY]\n{history_text}\n\n"
        f"[STUDENT QUESTION]\n{question}\n\n"
        "[COACH ANSWER]\n"
    )
    return pipe((prompt, img or st.session_state.image), gen_config=QA_GEN).text.strip()

# level starts from 1
def pick_hint(hints, level):
    if not hints:
        return "No hint available yet. Try to generalize your condition and check the loop boundaries."
    idx = max(0, min(level - 1, len(hints) - 1))
    return hints[idx]

def render_image_html(img_bytes: bytes, height_px: int = 400, caption: str = "Current block-code image"):
    b64 = base64.b64encode(img_bytes).decode()
    html = f"""
    <div style="height:{height_px}px; display:flex; justify-content:center; align-items:center; overflow:hidden; border-radius:8px;">
        <img src="data:image/png;base64,{b64}" style="max-height:{height_px-20}px; width:auto;">
    </div>
    <div style="text-align:center; color:#6b7a99; font-size:12px; margin-top:4px;">{caption}</div>
    """
    return html

# =================== GUI ===================
st.set_page_config(page_title="VLM Tutor (GUI)", layout="wide")

if "history_text" not in st.session_state: st.session_state.history_text = ""
if "assessment_text" not in st.session_state: st.session_state.assessment_text = ""
if "hints" not in st.session_state: st.session_state.hints = []
if "level" not in st.session_state: st.session_state.level = 1
if "problem" not in st.session_state: st.session_state.problem = PROBLEM
if "img_bytes" not in st.session_state: st.session_state.img_bytes = None
if "inited" not in st.session_state: st.session_state.inited = False
if "image" not in st.session_state: st.session_state.image = load_image(IMAGE_PATH)

left, right = st.columns([7,5], gap="large")

# left area
with left:
    st.subheader("Block Code Area")
    with st.container(border=True):
        preview = st.empty()
        def update_preview():
            preview.empty()
            if st.session_state.img_bytes:
                html = render_image_html(st.session_state.img_bytes, height_px=400)
                preview.markdown(html, unsafe_allow_html=True)
            else:
                preview.markdown(
                    "<div style='height:400px;display:flex;align-items:center;"
                    "justify-content:center;color:#6b7a99;background:#f5f8ff;border-radius:8px;'>"
                    "Upload an image below to preview it here</div>",
                    unsafe_allow_html=True
                )
        update_preview()

        uploaded = st.file_uploader("Upload block-based code image (png/jpg)",
                                    type=["png","jpg","jpeg"])
        st.session_state.problem = st.text_area("Problem statement",
                                                value=st.session_state.problem,
                                                height=70)

        colA, colB = st.columns([1,1])
        with colA:
            start = st.button("Assess (initialize)", type="primary", use_container_width=True)
        with colB:
            reset = st.button("Reset session", use_container_width=True)

        if uploaded:
            st.session_state.img_bytes = uploaded.read()
            update_preview()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(st.session_state.img_bytes); tmp.flush()
            st.session_state.image = load_image(tmp.name)

        if start:
            if st.session_state.img_bytes is None:
                st.warning("Please upload an image first.")
            else:
                assessment_text, hints = assess_and_hints(
                    st.session_state.problem,
                    st.session_state.image,
                    history_text=st.session_state.history_text
                )
                st.session_state.assessment_text = assessment_text
                st.session_state.hints = hints
                st.session_state.level = 1
                assess_body = assessment_text.split("[ASSESSMENT]")[-1].split("[HINTS]")[0].strip() \
                              if "[ASSESSMENT]" in assessment_text else assessment_text
                st.session_state.history_text += (
                    f"\n[T0] Initial code from image.\n"
                    f"[T0] Code assessment:\n{assess_body}\n"
                    f"[T0] Coach hint (level 1): {pick_hint(hints,1)}\n"
                )
                st.session_state.inited = True

        if reset:
            st.session_state.history_text = ""
            st.session_state.assessment_text = ""
            st.session_state.hints = []
            st.session_state.level = 1
            st.session_state.problem = PROBLEM
            st.session_state.img_bytes = None
            st.session_state.inited = False
            st.session_state.image = load_image(IMAGE_PATH)
            st.rerun()

# right area
with right:
    st.subheader("Chat Interface")
    with st.container(border=True):
        if not st.session_state.inited:
            st.info("Upload an image and click **Assess (initialize)** to start.")
        else:
            # initialize hint override status
            if "last_hint_override" not in st.session_state:
                st.session_state.last_hint_override = None

            # "More specific" button
            col_hint, col_btn = st.columns([3, 1])
            with col_hint:
                st.markdown(f"**[Coach Hint] (level={st.session_state.level})**")

                current_hint = (
                    st.session_state.last_hint_override
                    if st.session_state.last_hint_override is not None
                    else pick_hint(st.session_state.hints, st.session_state.level)
                )
                st.markdown(f"<div class='box'>{current_hint}</div>", unsafe_allow_html=True)

            with col_btn:
                if st.button("More specific", type="secondary", use_container_width=True):
                    if st.session_state.level < 3:
                        st.session_state.level += 1
                        st.session_state.last_hint_override = None
                        hint = pick_hint(st.session_state.hints, st.session_state.level)
                    else:
                        hint = ("If you are curious about more information, please ask questions.")
                        st.session_state.last_hint_override = hint

                    st.session_state.history_text += (
                        f"[T] Student requests more specific hint.\n"
                        f"[T] Coach hint (level {st.session_state.level}): {hint}\n"
                    )
                    st.rerun()

            # three tabs below
            tab1, tab3, tab4 = st.tabs([
                "(1) Revise code",
                "(2) Ask a question",
                "(3) Code assessment"
            ])

            with tab1:
                revised = st.text_area("Paste your revised code here:", height=140)
                if st.button("Submit revised code", type="primary", use_container_width=True):
                    if revised.strip():
                        st.session_state.history_text += f"[T] Student revised code:\n{revised}\n"
                        assessment_text, hints = assess_and_hints(
                            st.session_state.problem,
                            st.session_state.image,
                            history_text=st.session_state.history_text,
                            revised_code_text=revised
                        )
                        st.session_state.assessment_text = assessment_text
                        st.session_state.hints = hints
                        st.session_state.level = 1
                        st.session_state.last_hint_override = None
                        assess_body = assessment_text.split("[ASSESSMENT]")[-1].split("[HINTS]")[0].strip() \
                                      if "[ASSESSMENT]" in assessment_text else assessment_text
                        st.session_state.history_text += (
                            f"[T] Code assessment:\n{assess_body}\n"
                            f"[T] Coach hint (level 1): {pick_hint(hints,1)}\n"
                        )
                        st.rerun()

            with tab3:
                q = st.text_area("Type your question:", height=100)
                if st.button("Ask", type="primary", use_container_width=True):
                    if q.strip():
                        st.session_state.history_text += f"[T] Student asks: {q}\n"
                        ans = answer_question(q, st.session_state.history_text, img=st.session_state.image)
                        st.markdown("**[Coach Answer]**")
                        st.markdown(f"<div class='box'><pre>{ans}</pre></div>", unsafe_allow_html=True)
                        st.session_state.history_text += f"[T] Coach answers: {ans.strip()}\n"

            with tab4:
                assess_body = st.session_state.assessment_text.split("[ASSESSMENT]")[-1].split("[HINTS]")[0].strip() \
                              if "[ASSESSMENT]" in st.session_state.assessment_text else st.session_state.assessment_text
                st.markdown("**[Code Assessment]**")
                st.markdown(f"<div class='box'><pre>{assess_body}</pre></div>", unsafe_allow_html=True)
