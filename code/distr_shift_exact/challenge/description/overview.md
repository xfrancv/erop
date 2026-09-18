A classifier is trained on one mix of classes and deployed on another. It is
still a good classifier — it just has the wrong idea about how common each class
is, and it has no labels to correct itself with.

In this competition you are given such a classifier's **output** — eight class probabilities per image, not the pixels — and test images arriving in **batches**. Every batch was drawn under its own class mix, unknown to you but guaranteed to be one of eight possibilities you are told in advance. A batch of 100 images says a lot about which mix it came from; a batch of 1 says almost
nothing.

**Goal:** for every test image, predict its class **and** attach a confidence
score. We keep the 80 % of your predictions you were most confident about, and
measure how much worse they are than a reference predictor that was told each
batch's true class mix. Lower is better.

The interesting part is the confidence, not the labels. Two batches can be
equally hard to classify while differing completely in how well you could pin
down their class mix, and the scoring is built so that noticing the difference
pays.
