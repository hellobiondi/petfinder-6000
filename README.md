# PetFinder 6000

## Description
A multi-modal recommender system hosted (was) on Amazon Web Services (AWS) that recommends users cats from Cat Welfare Society they would most likely adopt.

## Recommender System

In this series of screenshots, you can find some EDA on our collected dataset, our methodology when approaching the problem and final results.

![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss1.png)
![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss2.png)
![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss3.png)
![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss4.png)
![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss5.png)
![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss6.png)
![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss7.png)
![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss8.png)

## Cloud Architecture & MLOps

![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss9.png)
![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss10.png)
![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss11.png)
![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss12.png)
![Screenshot](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss13.png)

## File structure
| Folder           | Details                                                                                                                                       |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| model            | Notebooks for model training, hyperparameter tuning, evaluation and inference. Includes Dockerfiles for custom training and inference images. |
| pre-processing   | Notebooks for data pre-processing                                                                                                             |
| pipelines        | Scripts and notebooks for creating processing, training and deployment pipelines                                                              |
| process_new_user | Scripts and notebook for creating lambda function to pull generated rankings from S3                                                          |

## Team
This project was done with my teammates, Ruo Xi, Shu Xian, Jun Yi and Adrian in fulfilment of our MITB Programme (Artificial Intelligence).
Notable libraries used were: Cornac, recommenders
