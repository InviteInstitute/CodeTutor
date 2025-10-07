# 🧠 TutorVLM: Streamlit Interface for Block-Based Programming Feedback

TutorVLM is an interactive **Visual-Language Tutor** built upon the [InternVL3-14B](https://huggingface.co/OpenGVLab/InternVL3-14B) model.  
It helps students learn **block-based programming** (e.g., Scratch-style code) by assessing their code,  
providing stepwise hints, and answering natural language questions — all through a Streamlit GUI.

---

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourname/tutorSys.git
   cd tutorSys

## 🧩 Environment Setup

Use the following commands to create and activate the environment:

   ```bash
   # Create the environment (this may take several minutes)
   conda env create -f tutorVLM_env.yaml
   
   # Activate it
   conda activate tutorVLM

## 🖥️ Run the Interface

Once the environment is activated, launch the interactive tutor GUI:

   ```bash
   # Run the main Streamlit interface
   streamlit run code_tutor_interface.py
