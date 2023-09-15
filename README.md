# PetFinder 6000

## Description
A multi-modal recommender system hosted (was) on Amazon Web Services (AWS) that recommends users cats from Cat Welfare Society they would most likely adopt.

## Recommender System
This portion outlines the insights gleaned from the dataset, methodologies employed, model performance and some sample results.

### The Application
![App preview](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss1.png)

### Exploratory Data Analysis
![Adopter attributes](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss2.png)
![Cat attributes](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss3.png)
![Power law at play](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss4.png)

### Methodologies
![Metrics used](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss8.png)
![Cold-start](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss6.png)

### Results
![Model performance](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss5.png)
![Sample results](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss7.png)

## Cloud Architecture & MLOps
This portion outlines the cloud architecture and pipelines that was deployed on AWS.

### ML Lifecycle and pipelines (Zoomed out)
![General architecture](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss9.png)
![Pipelines overview](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss10.png)

### Pipelines (Granular)
![Data collection & preparation](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss11.png)
![Model training](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss12.png)
![Rank generation](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss13.png)

## File structure
| Folder           | Details                                                                                                                                       |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| model            | Notebooks for model training, hyperparameter tuning, evaluation and inference. Includes Dockerfiles for custom training and inference images. |
| pre-processing   | Notebooks for data pre-processing                                                                                                             |
| pipelines        | Scripts and notebooks for creating processing, training and deployment pipelines                                                              |
| process_new_user | Scripts and notebook for creating lambda function to pull generated rankings from S3                                                          |

## Team
This project was done with my teammates, Ruo Xi, Shu Xian, Jun Yi and Adrian in fulfilment of our MITB Programme (Artificial Intelligence), and I could never have done it without them!
Notable libraries used were: Cornac, recommenders
