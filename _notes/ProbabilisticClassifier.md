# Probabilistic Classifiers

## Bayes Classifier

* Simple Explanation given [here](http://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification)

* **Conditional Probability** : What is the probability of event E occurring given some other event D has already happened?
	- Let's say that there is some Outcome O. And some Evidence E. From the way these probabilities are defined: The Probability of having both the Outcome O and Evidence E is: (Probability of O occurring) multiplied by the (Prob of E given that O happened)
	- Evidence and Outcome

* **Naive Bayes** : we have to predict an outcome given multiple evidence. In that case, the math gets very complicated. To get around that complication, one approach is to 'uncouple' multiple pieces of evidence, and to treat each of piece of evidence as independent. This approach is why this is called naive Bayes.

> P(Outcome/Multiple Evidence) = P(Evidence1/Outcome) x P(Evidence2/outcome) x ... x P(EvidenceN/outcome) x P(Outcome) scaled by P(Multiple Evidence)

> P(outcome/evidence) = ( P(Likelihood of Evidence) x Prior prob of outcome ) / P(Evidence)
                    			
* Naive Bayes Classifier ultimately *reduces* to

P(Banana/evidence) = 1/z * Prob(Banana) x Prob(Evidence1/Banana).Prob(Evidence2/Banana)...

P(Orange/Evidence) = 1/z * Prob(Orange) x Prob(Evidence1/Orange).Prob(Evidence2/Orange)...

P(Other Fruit/Evidence) = 1/z * Prob(Other) x Prob(Evidence1/Other).Prob(Evidence2/Other)...

## MLE for Gaussian

* MLE is a parametric model, where you estimate parameters of a distribution, which maximizes the probability of the observation of data X

* Laplace smoothing?