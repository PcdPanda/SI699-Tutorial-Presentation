# SI699 Tutorial Project

[Slide deck](https://docs.google.com/presentation/d/1YiUaIcKpRbHQw0kBXYqOCc7F9aRvCc19_MkJoIRTO1Y/edit#slide=id.g2017ecfacf7_0_53)

## 1. Dataset

Original dataset from the [repo](https://github.com/luxuan09/emoji_predicts_dropouts/tree/master/data) of the paper

| Filename                               | Description                                           |
| -------------------------------------- | ----------------------------------------------------- |
| actor_emojis                           | Emoji count in 264808 posts                           |
| emoji_duplicate_mapping                | Map the duplicate emojis with different utf-8 code    |
| 2018_post_type_emojis                  | Emoji count in 6 kinds of posts                       |
| 2018_posttype_posts                    | Count of posts in 6 kinds of posts                    |
| 2018_posttype_emojiposts               | Count of posts with emojis in 6 kinds of posts        |
| 2018_repolang_posts                    | Count of posts in 324 languages                       |
| 2018_repolang_emojiposts               | Count of posts with emojis in 324 languages           |
| 2018_repolang_repos                    | Count of repos in 324 languages                       |
| 2018_repolang_emojis                   | Count of repos with emojis in 324 languages           |
| 2018_emoji_and_non_emoji_user_features | 529616 user info with 64 features                     |
| active\_(non\_)emoji_users             | two groups of 67664 users with emoji and dropout info |
| lang_emoji_dropout                     | dropout distribution among language                   |

## 2. Code Go-through

#### 2.1 Environment Setup

- Python version should be higher than 3.8.5

  ```sh
  pip install -r requirement.txt
  ```

#### 2.2 Descriptive Analysis

1. **Popularity Analysis**

   - There are 2699 emojis and the most popular 20 are 🚀 ✅ ⬆️ ☁️ 👍 ⬇️ 🎉 🌴 🎫 ✔️ ⚠️ ❌ ⌨️ 🚨 😄 ♻️ 📺 💛 ❤️ 🔥

   - The popularity among 6 post types

     | Post type            | Emojis              | Post with emoji count | Post with emoji count Percentage |
     | -------------------- | ------------------- | --------------------- | -------------------------------- |
     | Issues_comments      | 👍 ⌨️ 😄 🎉 🚀 💪 😉 🚨 📺 ✨ | 9436402               | 0.033                            |
     | Issues               | 🚨 📺 ❌ ⚠️ ✔️ 🌴 👋 ✅ ⌨️ 👽 | 18485203              | 0.034                            |
     | Pull_reivew_comments | 👍 😄 🤔 😉 ⚠️ 😅 🙂 😆 ✅ 💯 | 8366087               | 0.029                            |
     | Pull                 | ☁️ 🎫 🚀 🌴 🎉 🚦 ♻️ 📅 🔕 🎟️ | 802005                | 0.041                            |
     | Pull_comments        | ⬆️ 🚀 ✅ ⬇️ 👍 🎉 ✔️ ❌ ⚠️ 💛 | 11898784              | 0.14                             |
     | Commit comments      | ✅ 📜 👍 🔥 💯 😄 🦍 🎉 🔴 ®️ | 13863740              | 0.06                             |


2. **Emoji usage by programming languages**

   - The top 20 programming languages with emoji count and entropy are 
   
     | Programming Language | Emoji Count | Emoji Entropy |
     | -------------------- | ----------- | ------------- |
     | JavaScript           | 1854        | 3.98          |
     | Python               | 1408        | 3.41          |
     | Java                 | 1096        | 3.38          |
     | C++                  | 1250        | 3.68          |
     | Go                   | 891         | 3.38          |
     | HTML                 | 1593        | 3.73          |
     | PHP                  | 1190        | 4.17          |
     | Ruby                 | 1018        | 2.31          |
     | TypeScript           | 1001        | 3.38          |
     | C#                   | 800         | 3.46          |
     | C                    | 717         | 3.69          |
     | CSS                  | 1154        | 4.14          |
     | Shell                | 778         | 4.34          |
     | Rust                 | 606         | 3.57          |
     | Scala                | 635         | 2.76          |
     | Swift                | 741         | 3.55          |
     | Objective-C          | 523         | 3.47          |
     | PowerShell           | 341         | 1.09          |
     | Jupyter Notebook     | 388         | 3.41          |
     | Kotlin               | 459         | 3.6           |

#### 2.3 Feature Analysis

- Emoji usage varies in repositories that have different primary programming languages


  ![Emoji usage varies in repositories that have different primary programming languages](visualization/emoji_langs.png)

   - One should control for programming languages when analyzing the relation of emoji usage and work-related status and outcomes.

 - Working status measurement
  
     | Categories | Working Status |
     | -------------------- | ----------- |
     | D1           | Activity level        |
     | D2              | Trend of activities        |
     | D3                | Type of activities        |
     | D4                 |  Type of posts        |
     | D5                  | Time management         |

1. **Feature selection**
   - Use log and percentage method to create 64 features of users
   - Use Variance Inflation Factor(VIF) method to remove correlated features (VIF > 5), and only leaves 45 features in total to ensure explainability of the model

2. **Regression Analysis**
   - OLS Regression for Emoji Usage on working status: 
     - #Emoji Posts
     - prop. Emoji Posts
     - prop. Emotional Emoji
       - LIWC emoji score (positive, negative, anger, anxiety, sadness)
  
3. **Label Analysis**
   - Why use dropout rate as feature

#### 2.4 Prediction


1. **Model Setting**
2. 


