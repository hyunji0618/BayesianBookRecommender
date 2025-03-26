# **Book Recommendation System Using Bayesian Machine Learning**
**University of Chicago MS in Applied Data Science**

**Course:** Bayesian Machine Learning with Generative AI Applications

**Date:** 03/13/2025

**Contributors:**
- Sam Fisher  
- Daniel Sa  
- Jazil Karim  
- Amy (Hyunji) Kim  

## **1 Introduction** 

Traditional recommendation systems often rely on the most common items or collaborative filtering techniques, which group users and items based on similarily measures. While these methods can be effective, they struggle in scenarios where data is sparse or user preferences evolve over time, leading to suboptimal recommendations.

Bayesian collaborative filtering enhances recommendation quality by incorporating probability-based predictions. Rather than solely relying on similarity-based heuristics, this approach estimates the likelihood of a user enjoying an item based on observed ratings and latent factors. A key component of this model is the likelihood function, which captures the probability distribution of observed ratings based on user-item interactions. By leveraging Bayesian principles, the model introduces latent variables to capture hidden factors influencing user preferences and dynamically updates its predictions as new data becomes available, offering a more robust and flexible recommendation framework.

---

## **2 Project Summary**

### 2.1 Bayesian Recommendation System for Book Ratings

In this project, we develop a **Bayesian Recommendation System** to predict how users will rate books they have not yet seen. Unlike traditional matrix factorization techniques such as **Singular Value Decomposition (SVD)**, our approach leverages a **Bayesian Network (Bayes Net)** to incorporate user biases, item biases, and latent factors while also quantifying uncertainty in predictions.  

### 2.2 Model Overview  

Our model represents each user with a **latent preference vector (αᵤ)** and each book with a **latent attribute vector (βᵢ)**. Both are drawn from prior distributions that help  regularize the model and prevent overfitting. The predicted rating follows a **probabilistic distribution** which incorporates:  

- A **global rating mean**: Captures overall rating tendencies
- A **user bias term**: Reflects individual rating tendencies (How lenient or harsh a user typically rates books)
- A **book bias term**: Represents the general quality of a book across all users
- The **dot product of preference and attribute vectors**: Represents user-item interactions

### 2.3 Inference & Estimation  

To estimate model parameters, we employ **Markov Chain Monte Carlo (MCMC)** for robust posterior estimation. Additionally, we implement **Automatic Differentiation Variational Inference (ADVI)** as our final inference method to enhance computational efficiency and ensure adaptability to different datasets and user behaviors. All Bayesian parameters are learned directly from the data.  

### 2.4 Dataset 

The project utilizes the **Book Recommendation Dataset** sourced from Kaggle, which contains real-world book ratings and user interactions: [Dataset Link](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

**Origin & Content**

This dataset is derived from the **Book-Crossing dataset**, originally compiled by Cai-Nicolas Ziegler in 2004. Book-Crossing was an online community where users freely shared their book ratings and reviews.

The dataset provides a rich source of information for building a recommendation system including:

- User Information: Anonymized data about users, such as age and location.
- Book Metadata: Details about books, including titles, authors, publication years, and publishers.
- User Ratings: Explicit ratings given by users on a scale from 0 to 10. A rating of 0 indicates an implicit interaction (e.g., the user might have shown interest in the book without explicitly rating it).

With over **500,000 ratings from more than 100,000 users**, this dataset offers a substantial and diverse collection of user preferences and book characteristics. This allows for robust training of a recommendation model.

**Processed Dataset**

The project uses a structured version of the dataset, which includes the following fields:

- User-ID: A unique identifier for each user.
- ISBN: The International Standard Book Number, a unique commercial book identifier.
- Book-Rating: The rating given by the user to the book.

Here's a glimpse of the processed data:

| User-ID | ISBN       | Book-Rating |
|---------|------------|-------------|
| 276725  | 034545104X | 0           |
| 276726  | 0155061224 | 5           |
| 276727  | 0446605239 | 0           |
| 276729  | 052165615X | 3           |
| 276729  | 0521795028 | 6           |

This processed data will be pivotal in training our Bayesian model to accurately predict user preferences and provide personalized book recommendations.

### 2.5 Model Evaluation 

We evaluate the model’s performance using the following key metrics:  

**Precision:**

$$
Precision = \frac{\text{Correctly classified actual positives}}{\text{Everything classified as positives}}
$$

- Measures the proportion of correctly recommended items among all items classified as relevant

**Recall:**

$$
Recall = \frac{\text{Correctly classified actual positives}}{\text{All actual positives}}
$$

- Assesses how well the model captures all relevant items by measuring the proportion of actual positives that were successfully recommended.

**Root Mean Squared Error (RMSE):**

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

- Quantifies the accuracy of rating predictions by calculating the average deviation from actual ratings, with larger errors weighted more heavily.

Among these metrics, **precision** is particularly important in a recommendation system, as ensuring that positive predictions are accurate enhances user satisfaction and trust.

Overall, our Bayesian approach not only **enhances recommendation accuracy** but also provides a probabilistic framework that accounts for **uncertainty**. This allows the system to adapt dynamically to new data, improving personalization and trustworthiness in recommendations while mitigating the risks of incorrect predictions.

### 2.6 Advantages of Bayesian Collaborative Filtering 

Bayesian collaborative filtering offers several advantages over traditional filtering approaches:

1. **Handling Uncertainty**  
   Bayesian methods excel at managing uncertainty by incorporating prior knowledge. They dynamically update predictions as new data becomes available, making them particularly useful in environments where user preferences change frequently.

2. **Addressing the Cold Start Problem**  
   Bayesian models can better handle the cold start problem by utilizing prior distributions based on historical data or demographic information. This allows the system to make educated and improved recommendations for new users and items.

3. **Integration of Prior Knowledge**  
   These methods allow for the incorporation of domain-specific prior knowledge, making them especially useful when data is limited. This leads to more informed and accurate recommendations.

4. **Flexibility and Adaptability**  
   Bayesian models are adaptable to various data types and recommendation scenarios. They can seamlessly integrate multiple data sources, including user behavior, item attributes, and contextual factors, leading to more relevant recommendation.

5. **Probabilistic Framework**  
   By providing a probabilistic framework, Bayesian methods quantify uncertainty in recommendations, leading to more interpretable and reliable suggestions.

6. **Scalability**  
   Bayesian models can scale effectively with large datasets, making them more suitable for real-world applications compared to traditional collaborative filtering methods, particularly when dealing with large datasets.

7. **Enhanced Personalization**  
   By capturing user-specific latent factors, Bayesian approaches offer highly tailored recommendations that adapt to individual preferences.

By leveraging Bayesian collaborative filtering, recommendation systems can achieve greater adaptability, reliability, and personalization, ultimately improving user experience across various applications.

---

## **3 Theory: Collaborative Filtering Model** 

This section describes a collaborative filtering model designed to predict user ratings for books. The model leverages past user-book interactions to suggest books to users.

### 3.1 Equation for Predicting Ratings

The predicted rating $r_{u,i}$ (the rating user $u$ gives to item $i$) is modeled as a gamma poisson:

$$
r_{u, i} \sim \mathcal{GammaPoisson}\left(\mu+b_{u}+b_{i}+\alpha_{u}^{T} \beta_{i}, \sigma^{2}\right)
$$

Where the mean of the distribution is the sum of the following components:

*   $\mu$: The global average rating across all users and books. This represents the overall popularity or quality of books in general.

*   $b_{u}$: The user bias. This term captures the tendency of a user to give consistently higher or lower ratings than average.

*   $b_{i}$: The item (book) bias. This term captures the tendency of a book to receive consistently higher or lower ratings than average.

*   $\alpha_{u}^{T} \beta_{i}$: The dot product of the user preference vector $\alpha_{u}$ and the book feature vector $\beta_{i}$. This term captures the interaction between specific user preferences and specific book characteristics.  $\alpha_u$ represents the latent (hidden) preferences of user $u$, while $\beta_i$ represents the latent features of book $i$. The dot product captures how well the user's preferences align with the book's features.

The variance of this normal distribution is $\sigma^{2}$, representing the overall noise or variability in the ratings.

### 3.2 Priors on Model Parameters  

To regularize the model and prevent overfitting, prior distributions are placed on the model parameters:

*   **Global Rating Prior:**

    *   $\mu \sim \mathcal{N}(0,5)$
    *   The global rating $\mu$ follows a normal distribution with a mean of 0 and a variance of 5.
    * This prior allows the model to learn an appropriate global rating, while regularizing it towards zero.

*   **User Biases:**

    *   $b_{u} \sim \mathcal{N}(0,1)$ for all users
    *   Each user bias $b_{u}$ follows a normal distribution with a mean of 0 and a variance of 1.
    * This prior encourages user biases to be small, preventing individual users from unduly influencing the overall rating predictions.

*   **Book Biases:**

    *   $b_{i} \sim \mathcal{N}(0,1)$ for all books
    *   Each book bias $b_{i}$ follows a normal distribution with a mean of 0 and a variance of 1.
    * This prior encourages book biases to be small, preventing individual books from unduly influencing the overall rating predictions. Some books might get consistently good or bad ratings, and this allows the model to capture that.

*   **Latent Book Features:**

    *   $\beta_{i} \sim \mathcal{N}(0, I \sigma_{\beta}^{2})$
    *   Each book feature vector $\beta_{i}$ follows a normal distribution centered at zero with identity covariance matrix scaled by $\sigma_{\beta}^{2}$.

*   **Latent User Preferences:**
    *   $\alpha_{u} \sim \mathcal{N}(0, I \sigma_{\alpha}^{2})$
    *   Each user preference vector $\alpha_{u}$ follows a normal distribution centered at zero with identity covariance matrix scaled by $\sigma_{\alpha}^{2}$.

### 3.3 Summary
This model predicts user ratings by combining a global average rating with user-specific and book-specific biases, as well as a term that captures the interaction between user preferences and book features. Prior distributions are used to regularize the model parameters for better generalization and improved recommendation accuracy by preventing overfitting.

---
## **4 Bayesian Recommendation Modeling** 

###  4.1 Data Preparation for Probabilistic Modeling

The dataset consists of user-book interactions, represented by ratings. We extract key components:  

- **Number of unique users**: `num_users`  
- **Number of unique books**: `num_books`  
- **User IDs**: `user_ids`  
- **Book IDs**: `book_ids`  
- **Ratings given by users**: `ratings`  

A hyperparameter, **latent dimension (latent_dim)**, is set to **2** to control the complexity of latent user and book factors.  

To adjust priors based on sparsity:  

- **User rating counts**: Number of books each user has rated  
- **Book rating counts**: Number of ratings each book has received  


### 4.2 Bayesian Model Initialization

The rating prediction is based on:  

- **Global mean rating (`mu`)**: Modeled with a Gamma prior (α=2, β=0.5)  
- **User bias (`user_bias`)**: Normally distributed with mean **0** and variance **1 / sqrt(user_rating_count)**  
- **Book bias (`book_bias`)**: Modeled similarly to user bias but for books  

**Hierarchical Priors for Latent Factors:**

Bayesian models use **hierarchical priors** to capture uncertainty at multiple levels.  

- **Standard deviation priors**:  
  - `sigma_u` (user factors) ~ Half-Cauchy(β=1)  
  - `sigma_b` (book factors) ~ Half-Cauchy(β=1)  

- **Latent factor distributions**:  
  - `user_factors` ~ Normal(0, `sigma_u`) → Shape: (num_users, latent_dim)  
  - `book_factors` ~ Normal(0, `sigma_b`) → Shape: (num_books, latent_dim)  

These factors represent underlying user preferences and book attributes.  


### 4.3 Rating Prediction

The predicted rating (`lambda_rating`) is computed as:
the combination of user biases, book biases, and latent factors.
$$
\lambda_{rating} = \exp(\mu + user\_bias[user] + book\_bias[book] + (user\_factors[user] \cdot book\_factors[book]))
$$

where:

- `mu` is the global average rating.
- `user_bias` and `book_bias` capture user-specific and book-specific biases in rating tendencies.
- `user_factors[user] ⋅ book_factors[book]` represents the interaction between user and book latent factors. This captures how much a user's preferences align with a book's characteristics.


### 4.4 Likelihood Function (Poisson Distribution)

Observed ratings (`ratings_obs`) follow a **Poisson likelihood**:

$$
ratings\_obs \sim \text{Poisson}(\lambda_{\text{rating}})
$$

This allows the model to learn parameters (user/book biases and latent factors) that best explain observed ratings.


### 4.5 Bayesian Inference & Model Training

The model parameters are estimated using **Bayesian inference**, which updates prior beliefs based on observed data to compute **posterior distributions**. We use:  

- **Markov Chain Monte Carlo (MCMC)** for robust posterior estimation  
- **Automatic Differentiation Variational Inference (ADVI)** as an alternative for faster convergence  


### 4.6 Model Updates & Learning

During training, the model iteratively updates:  

1. **Error Calculation**  
   - Example: `error = observed_rating - predicted_rating`  

2. **Parameter Updates**  
   - `user_bias[user] += learning_rate * error`  
   - `book_bias[book] += learning_rate * error`  
   - `user_factors[user] += learning_rate * error * book_factors[book]`  
   - `book_factors[book] += learning_rate * error * user_factors[user]`  

This process minimizes the difference between predicted and actual ratings.  

---


## **5 Multi-Step Lookahead & Bayesian Regret Minimization Strategy** 

Our recommendation system employs a **Bayesian regret minimization strategy** that uses **multi-step lookahead** to optimize the selection of recommended books. This method balances the trade-off between **exploitation** (recommending books we are confident the user will enjoy) and **exploration** (suggesting books where the system is less certain but could learn valuable information).

For each user, we compute:

- **Expected reward**: The mean predicted rating for each book, based on posterior samples from our Bayesian model.
- **Uncertainty (variance)**: The variability in those predictions, which represents how uncertain the system is about the user’s potential rating of a book.
- **Regret**: The difference between the best possible expected reward and the expected reward for each book.

By combining these components, the system assigns an **exploration score** to each book:

$$
\text{Exploration Score} = \text{Expected Reward} + (\text{Exploration Factor} \times \text{Uncertainty})
$$

Books with **higher uncertainty** and **potential learning benefit** are prioritized when their **regret** exceeds a defined threshold. This lookahead mechanism ensures that some recommendations actively reduce uncertainty about user preferences, improving the model's long-term learning.

Ultimately, the **multi-step lookahead approach** selects a top-`k` list of books that balances immediate relevance with future learning potential, driving both short-term user satisfaction and long-term system improvement.

---

## **6 Results and Evaluaton**  

We assess our model's performance by evaluating the **predicted ratings** it generates. Due to computational constraints, the Bayesian model was trained using a subset of **1,000 samples** from the dataset, with **800 samples for training** and **200 samples for testing**. The evaluation was conducted on this subset to analyze the model’s effectiveness in making accurate predictions.

**Precision**

We define a threshold of **7** to classify the numerically predicted ratings as positive or negative and convert the ratings into a binary classification problem.

Precision measures the proportion of correctly predicted positive ratings among all predicted positive ratings.

  - **Training set (800 samples):** 60%
  - **Test set (200 samples):** 21%

**Recall**

Recall evaluates the proportion of actual positive ratings that were correctly identified by the model.

  - **Training set (800 samples):** 61%
  - **Test set (200 samples):** 20%

**Root Mean Squared Error (RMSE)**

RMSE quantifies the difference between predicted and actual ratings, with lower values indicating better prediction accuracy.

  - **Training set (800 samples):** 1.44
  - **Test set (200 samples):** 2.58

Our evaluation indicates that our Bayesian recommendation model performs well on the training data, though its precision and recall drop on the test set. The discrepancy suggests that the model may be overfitting to the training data, limiting the model’s ability to generalize effectively to new data.

Due to computational constraints, we trained our model on a subset of the dataset rather than the full dataset. We expect improved model performance when trained with the entire dataset.

Additionally, we aim to introduce more evlauation metrics to better assess the relevance of the top-k recommendations generated by our model.

## **7 Takeaways**

Our key takeaways include:

- Accurate priors are everything in Bayesian Machine Learning - well-defined priors significantly impact Bayesian model performance.
- Hyperparameter tuning should be the final step - optimizing other components first yields better results.
- Sampling can be more computationally expensive than you might think.
- Model prediction time is nothing to sneeze at! Computational efficiency should never be underestimated.

 ![Screenshot 2025-03-13 at 3.50.50 PM.png](data:image/png;base64,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)


## **8 Conclusions & Future Improvements** 
- Introduce variables of book and author metadata to improve predictions.
- Introduce variables of user metadata to improve predictions.
- Evaluate the recommendations our model provides to improve fine-tuning. Possible metrics that can be considered: Diversity, Coverage, NDCG, and MRR
- Utilize superior hardware to train better models (more latent dimensions, more data)


```python

```
