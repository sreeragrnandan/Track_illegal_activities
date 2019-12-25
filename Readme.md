# Hi, this is a Illegal_activity tracker project
Data for the processes <a hreh="https://github.com/sreeragrnandan/Data_Track_illegal_activities">[Click here](https://github.com/sreeragrnandan/Data_Track_illegal_activities)</a>
<br /> We are planing to make a system that will report illegal activities like smoking in public place, child begging/labour and 
women haresment
## Project Stage This Far
1) Made model for smoking issue (check whether a person is smoking or not) using Max Pooling technique (If there is a better technique    please correct us) 
<br />2)Data collection in progress (we are collecting as much as data we can) please visit the above link to see the datas
### Apporach 
We are using TensorFlow to construct the model(neural network)
#### Data into two Train and Validation
First the data is splitted into two Train and Validation
```python
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
# Directory with our training smoking pictures
train_smoking_dir = os.path.join(train_dir, 'smoking')
# Directory with our training not_smoking pictures
train_not_smoking_dir = os.path.join(train_dir, 'not_smoking')
# Directory with our validation smoking pictures
validation_smoking_dir = os.path.join(validation_dir, 'smoking')
# Directory with our validation not_smoking pictures
validation_not_smoking_dir = os.path.join(validation_dir, 'not_smoking')
```
#### Data vissualization
We are using matplotlib to make verify that Image is correctly organized
```python
for i, img_path in enumerate(next_smoking_pix+next_not_smoking_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)
```
<img src="img.jpg" height="250px" >
