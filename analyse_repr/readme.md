# How to use
As I developed this very incrementally it is not very usable, here is shown how the tool can be used to create plots.

## Requirments
* python 3
* libraries:
	* matplotlib
	* numpy
	* jenkspy
* The analyses data:
	* https://github.com/Max216/ThesisPKinDL/blob/master/analyse_repr_data/snli_1000_sent_activations.txt

## Different types of labeling
There are different commands to label words in different ways:
* ```positional``` labels the word based on the position within the sentence from *0* to *length - 1*
* ```simple_pos``` labels the word by POS tag, NN*, JJ* etc are combinded to a single label
* ```verb_pos``` labels the word by POS tag, only distinguish between all VB\* forms, the rest is OTHER
* ```nn_jj_pos``` same as above, but with JJ\* and NN\*
* ```mcw``` labels the most comen words (except ".") as the word, the rest OTHER
* ```words``` labels only a given set of words as the words, the rest OTHER
	* The set of words is given by appending ```--w "word1 word2 word3"```, seperated with whitespace
	* Example: ```$ python analyse.py <path> words --w "boy boys girl girls"```
	* You can combine several words into one label using ```--g "male=boy boys; female=girl girls"``` instead of using ```--w```. This way all *boy, boys* are labeled *male* and all *girl, girls* are labeled *female*. The remaining words are labeled OTHER.
	* All words in sentences are lowercased when checking for the specified word lists
* ```simple_dep``` labels the word as dependency label. Only uses a small subset of all labels, the rest is OTHER
	* You can also create labels like *\<Dependencylabel\>-lemma* by running this command with ```--lemma "lemma1 lemma2 ..."```
	* And combine several lemmas into one label by running ```--lemma "label1=lemma1 lemma2; label2=lemma3 lemma4"``` (like in the ```words``` command)
	* Since I included lemmas late, only here the list uses lemmas, in ```words``` the lists are lowercased token-based, not lemma-based.
* ```pp``` labels using a POS-pattern, using the simplified POS tags (or should, I don't know how good it really works.) But it has some positional information as well. I only really tried it out with one POS tag. 
	* Use it like ```$ python analyse.py <path> pp --pos_pattern <POS_TAG> ``` and it will label whether it is the *first, last intermediate* or *only* occurence of <POS_TAG> within the sentence. This is very slow though.

I only have a fixed amount of colors, the script might break or repeat colors when too many labels are used.

## Find interesting dimensions

### General 
By running 
```
$ python analyse.py <path_to_snli_1000_sent_activations.txt> <label_type>
```
three different plots are created like https://github.com/Max216/ThesisPKinDL/blob/master/analyse_repr_data/Lexical%20Analysis:%20male%20female%20OTHER%5Bfilter%3D%5B'male'%2C%20'female'%5D%5D%20of%20300%20dimensions%20(most%20SD).png . They show for the first 300 sentences on what dimension what label was responsible for the value within the representation. Those plots are saved within the ./plots directory. Each plot shows the 300 sentences over 300 dimensions:
* First 300 dimensions
* 300 dimensions with the most standard deviation
* 300 dimensions with most values coming from one single label

### Find and Filter
You can also find labels you are intersting in by using ```--find LABEL``` and only consider sentences containing a label by using ```--filter LABEL```.

Example:
```
$ python analyse.py <snli_1000_sent_activations.txt> simple_pos --find JJ --filter JJ
```
This will create a plot with dimensions ordered by the amount of values coming from an adjective and is only showing sentences that contain an adjective.

### Find and filter a group
When labeling using ```words``` you might be interested in dimensions that have many values from the specified words but not necessarily only one of them. In this command you can append ```--group``` to order the dimensions by the amount of values coming from the specified words in total, not individually.

Example:
```
$ python analyse.py ./../../downloads/snli_1000_sent_activations.txt words --w "red green yellow blue" --group --filter "red green yellow blue"
```
This would create a plot with dimensions having the most values coming from one of those colors (Instead of looking for the most values coming ONLY from one of the colors). It is also filtered such that only sentences are shown that have one of the labels *red, green, blue, yellow*.

## Examine a dimenson
A dimension can be analysed using the ```--details <dimension number>``` command.
Example:
```
$ python analyse.py <path> simple_pos --details 757
```
This will create two plots of the dimension *757* labeled using the simplified POS tags.
* A bar chart will indicate for each label, how many times it was responsible for the value within the specified dimension
* Another chart will indicate for each label  the mean, max and min value in the dimension together with the standard deviation.
* The console output will be the tokens per label

You can also use ```--filter``` to create this plot based on a subset of all 1000 sentences only, if they contain the filter-label.

### Clustering
You can also divide the representation into different clusters having roughly similar values using ```--cluster <amount of clusters>```. this will create the same two plots as above, however now it is based on the clusters instead of the labels. The labels are shown in each bar.

Example:
```
python analyse.py <path> simple_pos --details 757 --cluster 3
```
* This will cluster dimension *757* into three clusters, each of these clusteres (named cat*\) are shown as a single bar.
* The console output shows the words per cluster (if a word is in several clusters it is listed below the most likely cluster)

### Histogram

You can create a histogram of one dimension by appending  ```--hist <bin size>```. This will divide the values reached in the given dimension using the specified dimension and shows how many words reached what value in the dimension. By appending ```--t <threshold>``` all words reaching a higher value than specified will be printed in the console.

Example:
```
$ python analyse.py <path> simple_pos --details 757 --hist 0.05 --t 0.5
```
This will show a histogram with a bin size of *0.05*. Each bar will be colored using the simplified POS labels that occur within the given bar. All words with a value greater of *0.5* within the representation are printed in the console.