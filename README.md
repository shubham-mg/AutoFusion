# AutoML Web App for Classification and Regression

This is a web application built using Streamlit that offers an automated machine learning (AutoML) solution for classification and regression tasks. Users can upload their datasets, perform Exploratory Data Analysis (EDA) using the `pandas_profiling` and `streamlit_pandas_profiling` libraries, and train machine learning models using the `PyCaret` library. The application has been deployed on Streamlit Sharing for easy access.

## Features

1. **Upload Your Dataset**: Users can upload their dataset in CSV format to get started with the analysis and modeling.

2. **Exploratory Data Analysis (EDA)**: The application generates an EDA report using `pandas_profiling` and displays it using the `streamlit_pandas_profiling` component. This helps users to quickly understand the key insights and statistics of their data.

3. **Model Training**: Users can choose between classification and regression tasks. The application uses `PyCaret` to automatically set up the data, compare different machine learning models, and select the best-performing one based on the chosen task.

4. **Model Comparison**: After training the models, the application displays a comparison of model performance metrics, allowing users to make an informed decision about selecting the best model.

5. **Model Download**: Once the best model is selected, users can download the trained model in pickle format (`.pkl`) for future use.

## How to Use

1. Open the [AutoML Web App](https://autofusion.streamlit.app/) (provide link to the deployed app on Streamlit Sharing).

2. On the sidebar, you'll find the navigation options: "Upload", "EDA", "Modelling", and "Download".

3. **Upload**: Click on "Upload" to upload your dataset in CSV format. The uploaded dataset will be saved locally for further analysis.

4. **EDA**: Choose "EDA" to perform Exploratory Data Analysis on the uploaded dataset. The application will generate an interactive EDA report for a comprehensive understanding of the data.

5. **Modelling**: Select "Modelling" to start the machine learning model training process. Choose between classification and regression tasks. The application will use `PyCaret` to set up the data, compare models, and save the best model.

6. **Download**: In the "Download" section, you can download the trained model (in `.pkl` format) that achieved the best performance during the modeling process.

## Technologies Used

- Streamlit: Used to create the interactive web interface for the application.
- pandas_profiling: Used to generate an EDA report for the dataset.
- streamlit_pandas_profiling: Streamlit component to display the EDA report.
- PyCaret: Utilized for automatic data setup, model comparison, and selection of the best model.
- Streamlit Sharing: Deployed the application on Streamlit Sharing for easy access.

## Getting Started

To run the application locally:

1. Clone this repository: `git clone [repository_url]`
2. Install the required packages: `pip install streamlit pandas_profiling pycaret`
3. Run the application: `streamlit run app.py`

## Future Enhancements

- Include more customization options for model training and hyperparameter tuning.
- Allow users to save and load previous analysis sessions.
- Incorporate additional data visualization libraries for richer insights.
- Expand the model selection beyond PyCaret's options.
- Improve the user interface for a more intuitive experience.

## Author

Shubham Gupta

## Acknowledgements

Special thanks to the creators of Streamlit, pandas_profiling, streamlit_pandas_profiling, and PyCaret for providing the tools that made this application possible.

