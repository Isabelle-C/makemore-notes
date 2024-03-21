words=open('names.txt','r').read().split()

b={}
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram]=b.get(bigram, 0) + 1
        #print(ch1,ch2)

chars = sorted(set(''.join(words)))
char_to_int = {c: i+1 for i, c in enumerate(chars)}
int_to_char = {i+1: c for i, c in enumerate(chars)}
char_to_int['.']=0
int_to_char[0]='.'
N = torch.zeros((27,27), dtype=torch.int32)