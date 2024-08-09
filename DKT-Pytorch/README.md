# py_dkt
Deep knowledge tracing based on pytorch

## RUN
`python main.py`

## Eval & Predict
`eval.ipynb`


## Reference code
https://github.com/chsong513/DeepKnowledgeTracing-DKT-Pytorch 
    * 这个代码写的比较原始。没有模块化，loss function也是没有用的nn BCELoss ，看起来比较麻烦
https://github.com/GreenD93/py_dkt
    * 这个好处就是模块化比较好，结构清晰，缺点是metrics部分没有用到现成的包，比如sklearn.metrics，没有算auc
    * 不过最后eval.ipynb那块暂时没看清楚

## Paper
https://web.stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf


## 我的一些感触
* my_eval.ipynb mytest_mian.ipynb为我调试过程中的记录
* 首先这个问题很清楚根据用户的输入(batch_size maxstep numofquestion*2 ) 预测下一个step所有题目作答的概率
* 尤其是自定义loss fuction哪一块，最好自己用手推一遍，为什么要进行matmul？为什么matmul之后对角线上就是对应的预测概率？
* 值得注意的是，本代码encode过程，one-hot encode 把qid和correct信息融合了，前numofq 代表正确作答1，后numofq代表错误作答也为1，最后巧妙的还原question seq 和 answer seq！
    * 因为我看https://github.com/bigdata-ustc/EduKTM 这里是利用% numofq 和// numofq 还原的，但是其实在// numofq，错误题目会被还原成1 ，而正确会被还原成0 ，需要 truth = 1- truth，不知道作者为什么没有写？因此便换用了下面哪个refrences

* 还有就是本人身体这段时间不太舒服，之前的阑尾炎手术后没复查，这几天一直热热的，可恶啊！！！！
    

## 踩坑记录
* 我陆陆续续看了很多的git 实现，主要权威的有暨南大学的pykt，但是用的是mask的方法来标记填充位置的，暂时没看懂逻辑
* 另外就是https://github.com/bigdata-ustc/EduKTM 这里写的还好其实，挺容易懂的，但是疑问就出现在他计算loss的的过程！！很奇怪个人感觉不太对，已经提了issues1
'''python
def process_raw_pred(raw_question_matrix, raw_pred, num_questions: int) -> tuple:
    questions = torch.nonzero(raw_question_matrix)[1:, 1] % num_questions ##torch.nonzero(raw_question_matrix)[1:, 1]表示raw_question_matrix中非0的位置，即用户的有效回答
    length = questions.shape[0]
    pred = raw_pred[: length]
    pred = pred.gather(1, questions.view(-1, 1)).flatten()
    truth = torch.nonzero(raw_question_matrix)[1:, 1] // num_questions #truth表示真实作答情况，0表示回答正确，1表示回答错误，与原始数据相反?
    # truth = 1 - truth#这里逻辑是不是写错了！！ 0表示回答正确，1表示回答错误，与原始数据相反?
    
    return pred, truth


'''