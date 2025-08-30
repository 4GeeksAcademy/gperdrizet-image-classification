# Image classification environment set-up

Using this fork will get you up and running in a codespace in just a few minutes.

## 1. Get a Kaggle account

The dataset is hosted on Kaggle. To download it you need a free account. It's easy to set up via the following link:

[Login or Register | Kaggle](https://www.kaggle.com/account/login?phase=startRegisterTab)

After setting up your account, you will need to verify with a phone number to use the Kaggle API. You can do this from the *settings* tab in the menu revealed by clicking your profile picture at the top right of any Kaggle page.

Once you are registered and logged in:

- Go to the [Dogs vs Cats competition](https://www.kaggle.com/competitions/dogs-vs-cats) page
- Go to the 'Data' tab
- Scroll down and click 'Join competition'


## 2. Running with GitHub codespace

Start by forking this repository as you normally would.


### 2.1. Generate a Kaggle API key

- From the kaggle homepage, click on your profile picture in the upper right
- Select 'Settings'
- Scroll down, under API, click 'Create New Token'
- Click 'Continue'
- Save the key file on you local machine

The contents of the file should look like this:

```json
{"username":"your-user-name","key":"a-bunch-of-letters-and-numbers"}
```

### 2.2. Add your Kaggle credentials to Codespace secrets

You can find codespace secrets under 'settings' -> 'secrets and variables'. Full instructions for using secrets with codespaces [here](https://docs.github.com/en/codespaces/managing-your-codespaces/managing-your-account-specific-secrets-for-github-codespaces)


### 2.3. Start a codespace

Start a codespace as you normally would. On start-up it will download the dataset from Kaggle for you. All of the notebooks in the repo have built in file handling logic to extract and organize the images for you. If you are interested in knowing how this is done, check out the `get_data.sh` script in the project root directory and the function `prep_data` in the `image_classification_functions` module.

## 4. Running on Kaggle with GPU

It is also possible to run this project on Kaggle with free GPU access. See full instructions [here](https://github.com/gperdrizet/ds-12/blob/main/pages/guides/kaggle_notebooks.md)
