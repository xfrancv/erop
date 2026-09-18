The evaluation section describes how submissions will be scored and how participants should format their submissions. You don't need a sub-title at the top; the page title appears above. Below is an example of a typical evaluation page.

---

Submissions are evaluated on [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.


## Submission File
For each ID in the test set, you must predict a probability for the TARGET variable. The file should contain a header and have the following format:

    ID,TARGET
    2,0
    5,0
    6,0
    etc.