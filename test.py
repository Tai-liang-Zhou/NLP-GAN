import jieba
import codecs
from pyltp import Segmentor
import opencc

t2s = opencc.OpenCC('t2s')
#a = cc.convert('五千萬')
#
s2t = opencc.OpenCC('s2t')
#b = tt.convert(a)



segmentor = Segmentor()
segmentor.load("./ltp_data_v3.4.0/cws.model")
#jieba.load_userdict('dir.txt')
words = []
with codecs.open('review_generation_dataset/train/new_train_data.csv', 'r', 'utf-8') as ask_f:
    for line in ask_f:
        line = line.split(",")
        line0 = segmentor.segment(t2s.convert(line[0]))
        words.append(("|".join(line0)))
        line1 = segmentor.segment(t2s.convert(line[1]))
        words.append(("|".join(line1)))
        
        
vocab = []
for word in words:
    word = word.split("|")
    for index in range(len(word)):
        if word[index] not in vocab:
            vocab.append(word[index])

with codecs.open('review_generation_dataset/train/new_vocab.txt', "a", "utf-8") as voc_f:
    for word in vocab:
        voc_f.write(s2t.convert(word) + "\n")
    
        
        
    
        print("0 :" + line[0] + "\n")
        line0 = segmentor.segment(line[0])
        words.append(("|".join(line0)))
        print ("|".join(line0))
        print("1 :" + line[1])
        line1 = segmentor.segment(line[1])
        print ("|".join(line1))
        # sentence = jieba.cut(line[0])
        # sentence = (" ".join(sentence))
        # sentence2 = jieba.cut(line[1])
        # sentence2 = (" ".join(sentence2))
        # print (sentence+','+ sentence2)
        # wrrit_data = sentence+','+ sentence2
        # write_positive_file = codecs.open('review_generation_dataset/train/123.csv', "a", "utf-8")
        # write_positive_file.write(wrrit_data)
#        print(line)

a = words[1]
b = a.split("|")
for word in range(len(b)):
    print(b[word])

with codecs.open('review_generation_dataset/train/123.csv', 'r', 'utf-8') as ask_f:
    a = []
    for line in ask_f:
        line = line.split(",")
        for index in range(len(line)):
                    line[index] = line[index].strip()
                    line[index] = line[index].strip('\ufeff')
        a.append(line)
        print(line)


if initial_state_attention:
      attns = attention(initial_state)
for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)

      inputs = [inp] + attns
      x = Linear(inputs, input_size, True)(inputs)
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(
            variable_scope.get_variable_scope(), reuse=True):
          attns = attention(state)
      else:
        attns = attention(state)

      with variable_scope.variable_scope("AttnOutputProjection"):
        inputs = [cell_output] + attns
        output = Linear(inputs, output_size, True)(inputs)
      if loop_function is not None:
        prev = output
      outputs.append(output)