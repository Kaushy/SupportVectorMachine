def f(x):
    if len(str(x)) < 0: return 1
    else: return 0

twitterDataFrame['InReplyToStatusID'].map(f)