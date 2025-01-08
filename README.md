The detection of diabetes at an early stage is critical for effective management and treatment. This study introduces an optimized machine learning-based approach to predict diabetes with high accuracy. Employing a dataset from the UCI Machine Learning Repository, we evaluate multiple algorithms, including Support Vector Machines (SVM), Random Forest (RF), and Artificial Neural Networks (ANN). Our results indicate that the proposed algorithm achieves an accuracy of 75.3%, surpassing traditional methods. The findings highlight the potential of machine learning in medical diagnostics.

Diabetes is a chronic condition affecting millions worldwide. Early detection can significantly reduce complications and healthcare costs. Traditional diagnostic methods rely on biochemical tests, which are time-consuming and costly. Recent advances in machine learning (ML) have demonstrated the potential to automate and enhance diagnostic accuracy. This research focuses on leveraging ML algorithms to build a predictive model for diabetes detection. The aim is to achieve high precision and reliability, thereby enabling early intervention and better patient management.

The increasing prevalence of diabetes necessitates innovative approaches for early detection. Automated systems powered by ML can serve as complementary tools to traditional methods, offering faster and more accessible diagnostics.

Data preprocessing steps included:
■ Handling Missing Values: Missing values were imputed using the mean of the respective feature.
■Feature Scaling: Data was normalized to a [0, 1] scale to ensure consistency and improve algorithm performance.
■Train-Test Split: The dataset was divided into 70% training and 30% testing subsets to validate model performance.

The dataset used is the Pima Indians Diabetes Dataset, widely utilized in diabetes research. It contains 768 records with 8 attributes:
▫ Pregnancies
▫ Glucose
▫ Blood Pressure
▫ Skin Thickness
▫ Insulin
▫ BMI
▫ Diabetes Pedigree Function
▫ Age

Data preprocessing steps included:
• Handling missing values using median imputation.
• Normalization of numerical features.
• Splitting the dataset into training (70%) and testing (30%) subsets.

The diabetes detection algorithm is based on machine learning. The models evaluated include:
•Logistic Regression
•Random Forest Classifier
•Support Vector Machine (SVM)
•Neural Networks

The following metrics were used to assess the performance:
▫Accuracy
▫Precision
▫Recall
▫F1-Score
▫Area Under the Receiver Operating Characteristic Curve (AUC-ROC)
