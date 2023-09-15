# PetFinder 6000

# Table of Contents

1. [Description](#description)
2. [Recommender System](#recommender-system)
   - [The Application](#the-application)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
   - [Methodologies](#methodologies)
   - [Results](#results)
3. [Cloud Architecture & MLOps](#cloud-architecture--mlops)
   - [ML Lifecycle and pipelines (Zoomed out)](#ml-lifecycle-and-pipelines-zoomed-out)
   - [Pipelines (Granular)](#pipelines-granular)
4. [File Structure](#file-structure)
5. [Team](#team)

## Description
A multi-modal recommender system hosted (was) on Amazon Web Services (AWS) that recommends users cats from Cat Welfare Society they would most likely adopt.

## Recommender System
This portion outlines the insights gleaned from the dataset, methodologies employed, model performance and some sample results.

### The Application
![App preview](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss1.png)
*The app was first designed using Amplify Studio and then deployed subsequently on AWS Amplify. We had 404 cat profiles that had to manually scraped and cropped (this was especially painful)*

[Back to top](#table-of-contents)

### Exploratory Data Analysis
![Adopter attributes](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss2.png)
*Attributes of the adopters that registered on the app, as boxplots*

![Cat attributes](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss3.png)
*Attributes of the cats that were scraped from Cat Welfare Society, as boxplots*

![Power law at play](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss4.png)
*That's the Power law at play here! You can see the sharp drop-off in interaction, making data sparse (usual RecSys shenanigans)*

[Back to top](#table-of-contents)

### Methodologies
![Metrics used](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss8.png)
*Other than the usual F1, NDCG, NCRR, 'Distributional coverage' and 'Serendipity' was also implemented with understanding from:
https://eugeneyan.com/writing/serendipity-and-accuracy-in-recommender-systems/*

![Cold-start](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss6.png)
*To combat cold-start issues where we could not recommend effectively to new users who have not rated any cats, we retrieved embeddings from users and searched for another existing user with high cosine similarity as the cold-start user and showed them the same recommendations until they have rated some cats.*

[Back to top](#table-of-contents)

### Results
![Model performance](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss5.png)
*Across the models, it can be seen that 'vanilla' CF models such as WMF, BPR worked really well. But multi-modal models such as VBPR performed relatively well for our combined HarmonicMean metric, which was expected as people likely had strong visual preferences when it comes to cats. Text models did not work as well, which can be an indication that people tend to not pay heed to description as much as visuals.*
![Sample results](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss7.png)
*A sample result for a random adopter.*

[Back to top](#table-of-contents)

## Cloud Architecture & MLOps
This portion outlines the cloud architecture and pipelines that was deployed on AWS.

[Back to top](#table-of-contents)

### ML Lifecycle and pipelines (Zoomed out)
![General architecture](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss9.png)
*General architecture of the ML Lifecycle*

![Pipelines overview](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss10.png)
*Overview of data pipelines*

[Back to top](#table-of-contents)

### Pipelines (Granular)
![Data collection & preparation](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss11.png)
*Data collection & Preparation*

![Model training](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss12.png)
*Model training*

![Rank generation](https://github.com/hellobiondi/petfinder-6000/raw/main/screenshots/ss13.png)
*Rank generation: This is where the magic happens.*

[Back to top](#table-of-contents)

## File structure
| Folder           | Details                                                                                                                                       |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| model            | Notebooks for model training, hyperparameter tuning, evaluation and inference. Includes Dockerfiles for custom training and inference images. |
| pre-processing   | Notebooks for data pre-processing                                                                                                             |
| pipelines        | Scripts and notebooks for creating processing, training and deployment pipelines                                                              |
| process_new_user | Scripts and notebook for creating lambda function to pull generated rankings from S3                                                          |

[Back to top](#table-of-contents)

## Team
This project was done with my teammates, Ruo Xi, Shu Xian, Jun Yi and Adrian in fulfilment of our MITB Programme (Artificial Intelligence), and I could never have done it without them!
Notable libraries used were: Cornac, recommenders

[Back to top](#table-of-contents)
