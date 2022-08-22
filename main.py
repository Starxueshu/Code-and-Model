# This is a sample Python script.
import pickle

#import joblib
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
#from sklearn.linear_model import LogisticRegression
import streamlit as st


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green® button in the gutter to run the script.
def app():
    # Title
    st.header("A novel model to predict early death among bone metastasis patients with unknown primary cancer using "
              "machine learning approaches: an external validated study")
    st.sidebar.title("Parameters Selection Panel")
    st.sidebar.markdown("Picking up parameters according to conditions")
    # Input bar 1
    #height = st.number_input("Enter Height")

    # Input bar 2
    #weight = st.number_input("Enter Weight")

    # Dropdown input
    #eyes = st.selectbox("Select Eye Colour", ("Blue", "Brown"))
    age = st.sidebar.slider("Age", 30, 100)
    liverm = st.sidebar.selectbox("Liver metastasis", ("No", "Unknown", "Yes"))
    lungm = st.sidebar.selectbox("Lung metastasis", ("No", "Unknown", "Yes"))
    Radiation = st.sidebar.selectbox("Radiation", ("No/Unknown", "Yes"))
    Chemotherapy = st.sidebar.selectbox("Chemotherapy", ("No/Unknown", "Yes"))


    # Input bar 1
    #x22 = st.number_input("Importance of clinical practice in accordance with law")

    # Input bar 2
    #x36 = st.number_input("Construction status of hospital rule of law")

    # If button is pressed
    if st.button("Submit"):
        # Unpickle classifier
        #clf = joblib.load("gbm_clf_final_round.pkl")
        with open('rf_clf_final_round.pkl', 'rb') as f:
            ensemble_clf = pickle.load(f)

            # Store inputs into dataframe
        x = pd.DataFrame([[age, liverm, lungm, Radiation, Chemotherapy]],
                         columns=["age", "liverm", "lungm", "Radiation", "Chemotherapy"])
        x = x.replace(["No", "Unknown", "Yes"], [1, 2, 3])
        x = x.replace(["No", "Unknown", "Yes"], [1, 2, 3])
        x = x.replace(["No/Unknown", "Yes"], [1, 2])
        x = x.replace(["No/Unknown", "Yes"], [1, 2])

        # Get prediction
        prediction = ensemble_clf.predict_proba(x)[0, 1]
        # Output prediction
        st.text(f"Probability of experiencing medical disputes: {'{:.2%}'.format(round(prediction, 5))}")
        if prediction < 0.711:
            st.text(f"Risk group: low-risk group")
        else:
            st.text(f"Risk group: High-risk group")
        if prediction < 0.711:
            st.markdown(f"Recommendations: Patients in the high-risk group may better be treated with radiotherapy alone, best supportive care, or minimal invasive techniques such as cementoplasty to palliatively alleviate pain since patients might not have enough time to recovery from surgery.")
        else:
            st.markdown(f"Recommendations: For patients in the low-risk groups, more invasive surgery, such as excisional surgery, and long-course radiotherapy could be recommended, because those patients might suffer from poor quality of life for a very long time if only palliative interventions were performed.")

    st.subheader('About the model')
    st.markdown('The web calculator was created according the model developed by the random forest approach. Internal validation: AUC: 0.796 (95% CI: 0.746-0.847); accuracy: 0.778. External validation: AUC: 0.748 (95% CI: 0.653-0.843); accuracy: 0.745. Risk stratification: based on the best cut-off value (0.711), patients in the high-risk groups were above two times more likely to suffer from early death than patients in the low-risk groups (P<0.001).')

if __name__ == '__main__':
    print_hi('PyCharm')
    #load()
    app()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# /Users/wdscl/Desktop/WorkSpace/ceshiweb.xlsx
