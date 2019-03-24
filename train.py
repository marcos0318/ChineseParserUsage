# The base class of the model that can do the
import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
import gensim
from gensim.models import KeyedVectors
import sys
from newDataLoader import DataLoader
from time import time
from tqdm import tqdm
import os
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine
import json
import pickle
import argparse



class Model:

    def __init__(self, data, args):
        # random initialize the weights

        self.data = data
        vocab_size = data.vocab_size
        self.vocab_size = data.vocab_size
        embed_size = self.embed_size = 300
        self.res = args.restrict
        self.weight = args.weight

        rel_length = args.relational_embedding_size



        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

            

            # The center embedding should be initialize during the training process by "assign"
            # Directly use const is not memory efficient, because the constant is saved multiple times in the memory
            # The trainable here is set for not trainable for now
            


            W_amod = tf.get_variable("amod_tranform_matrix", [rel_length, embed_size])
            W_nsubj = tf.get_variable("nsubj_tranform_matrix", [rel_length, embed_size])
            W_dobj = tf.get_variable("dobj_tranform_matrix", [rel_length, embed_size])


            self.center_emb = tf.get_variable("center_emb", [vocab_size, embed_size], trainable=args.center_trainable)
            self.amod_emb = tf.matmul(tf.get_variable("amod_emb", [vocab_size, rel_length]), W_amod)
            self.nsubj_emb = tf.matmul(tf.get_variable("nsubj_emb", [vocab_size, rel_length]), W_nsubj)
            self.dobj_emb = tf.matmul(tf.get_variable("dobj_emb", [vocab_size, rel_length]), W_dobj)


            W_amod_context = tf.get_variable("amod_context_tranform_matrix", [rel_length, embed_size])
            W_nsubj_context = tf.get_variable("nsubj_context_tranform_matrix", [rel_length, embed_size])
            W_dobj_context = tf.get_variable("dobj_context_tranform_matrix", [rel_length, embed_size])


            self.center_emb_context = tf.get_variable("center_emb_context", [vocab_size, embed_size], trainable=args.center_trainable)
            self.amod_emb_context = tf.matmul(tf.get_variable("amod_emb_context", [vocab_size, rel_length]), W_amod_context)
            self.nsubj_emb_context = tf.matmul(tf.get_variable("nsubj_emb_context", [vocab_size, rel_length]), W_nsubj_context)
            self.dobj_emb_context = tf.matmul(tf.get_variable("dobj_emb_context", [vocab_size, rel_length]), W_dobj_context)

            self.emb_placeholder = tf.placeholder(tf.float32, [vocab_size, embed_size])
            self.emb_init = self.center_emb.assign(self.emb_placeholder)

            self.emb_placeholder_context = tf.placeholder(tf.float32, [vocab_size, embed_size])
            self.emb_init_context = self.center_emb_context.assign(self.emb_placeholder_context)


            self.amod_restrict_emb = self.restrict(self.amod_emb)
            self.nsubj_restrict_emb = self.restrict(self.nsubj_emb)
            self.dobj_restrict_emb = self.restrict(self.dobj_emb)

            self.amod_restrict_emb_context = self.restrict(self.amod_emb_context)
            self.nsubj_restrict_emb_context = self.restrict(self.nsubj_emb_context)
            self.dobj_restrict_emb_context = self.restrict(self.dobj_emb_context)

            self.predicate_amod_ids = tf.placeholder(tf.int32, [None])
            self.argument_amod_ids = tf.placeholder(tf.int32, [None])
            self.argument_prime_amod_ids = tf.placeholder(tf.int32, [None])

            self.predicate_nsubj_ids = tf.placeholder(tf.int32, [None])
            self.argument_nsubj_ids = tf.placeholder(tf.int32, [None])
            self.argument_prime_nsubj_ids = tf.placeholder(tf.int32, [None])

            self.predicate_dobj_ids = tf.placeholder(tf.int32, [None])
            self.argument_dobj_ids = tf.placeholder(tf.int32, [None])
            self.argument_prime_dobj_ids = tf.placeholder(tf.int32, [None])

         
            self.add_pred_amod()
            self.add_pred_nsubj()
            self.add_pred_dobj()

            self.add_prediction()
            self.add_loss()
            self.add_optimize()

    def get_tensor_shape(self, x, dim):
        return x.get_shape()[dim].value or tf.shape(x)[dim]

    def add_prediction(self):

        self.accuracy_amod = tf.reduce_mean(tf.to_float(tf.greater(self.pred_amod, self.pred_amod_prime)))
        self.accuracy_nsubj = tf.reduce_mean(tf.to_float(tf.greater(self.pred_nsubj, self.pred_nsubj_prime)))
        self.accuracy_dobj = tf.reduce_mean(tf.to_float(tf.greater(self.pred_dobj, self.pred_dobj_prime)))

    def restrict(self, embeddings):
        num_examples = self.get_tensor_shape(embeddings, 0)
        embedding_norms = tf.norm(embeddings, axis=1)
        mask = tf.greater(embedding_norms, tf.ones([num_examples]) * self.res)
        normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1) * self.res
        return tf.where(mask, x=normalized_embeddings, y=embeddings)

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
        return log_norm - marginalized_gold_scores  # [k]

    def add_pred_amod(self):
        # center embeddings
        predicate_embedding = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb, self.predicate_amod_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.amod_emb, self.predicate_amod_ids))
        argument_embedding = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb, self.argument_amod_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.amod_emb, self.argument_amod_ids))
        argument_prime_embedding = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb,
                                                          self.argument_prime_amod_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.amod_emb, self.argument_prime_amod_ids))

        # context embeddings
        predicate_embedding_context = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb_context, self.predicate_amod_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.amod_emb_context, self.predicate_amod_ids))
        argument_embedding_context = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb_context, self.argument_amod_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.amod_emb_context, self.argument_amod_ids))
        argument_prime_embedding_context = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb_context,
                                                          self.argument_prime_amod_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.amod_emb_context, self.argument_prime_amod_ids))

        self.overall_predicate_amod = predicate_embedding

        self.base_embedding_handler = tf.nn.embedding_lookup(self.center_emb, self.predicate_amod_ids)
        self.base_embedding_context_handler = tf.nn.embedding_lookup(self.center_emb_context, self.predicate_amod_ids)

        self.amod_embedding_handler = predicate_embedding
        self.amod_embedding_context_handler = argument_embedding_context


        # the score is calculated as pred * argu_context + argu * pred_context 
        self.pred_amod = tf.reduce_sum(tf.multiply(predicate_embedding, argument_embedding_context), 1) + \
        tf.reduce_sum(tf.multiply(predicate_embedding_context, argument_embedding), 1)

        self.pred_amod_prime = tf.reduce_sum(tf.multiply(predicate_embedding, argument_prime_embedding_context), 1) + \
        tf.reduce_sum(tf.multiply(predicate_embedding_context, argument_prime_embedding), 1)

    def add_pred_nsubj(self):
        predicate_embedding = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb, self.predicate_nsubj_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.nsubj_emb, self.predicate_nsubj_ids))
        argument_embedding = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb, self.argument_nsubj_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.nsubj_emb, self.argument_nsubj_ids))
        argument_prime_embedding = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb,
                                                          self.argument_prime_nsubj_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(self.nsubj_emb, self.argument_prime_nsubj_ids))

        # context embeddings
        predicate_embedding_context = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb_context, self.predicate_nsubj_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.nsubj_emb_context, self.predicate_nsubj_ids))
        argument_embedding_context = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb_context, self.argument_nsubj_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.nsubj_emb_context, self.argument_nsubj_ids))
        argument_prime_embedding_context = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb_context,
                                                          self.argument_prime_nsubj_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.nsubj_emb_context, self.argument_prime_nsubj_ids))

        self.overall_predicate_nsubj = predicate_embedding


        self.nsubj_embedding_handler = predicate_embedding
        self.nsubj_embedding_context_handler = argument_embedding_context

      

        self.pred_nsubj = tf.reduce_sum(tf.multiply(predicate_embedding, argument_embedding_context), 1) + \
        tf.reduce_sum(tf.multiply(predicate_embedding_context, argument_embedding), 1)

        self.pred_nsubj_prime = tf.reduce_sum(tf.multiply(predicate_embedding, argument_prime_embedding_context), 1) + \
        tf.reduce_sum(tf.multiply(predicate_embedding_context, argument_prime_embedding), 1)


    def add_pred_dobj(self):
        predicate_embedding = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb, self.predicate_dobj_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.dobj_emb, self.predicate_dobj_ids))
        argument_embedding = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb, self.argument_dobj_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.dobj_emb, self.argument_dobj_ids))
        argument_prime_embedding = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb,
                                                          self.argument_prime_dobj_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(self.dobj_emb, self.argument_prime_dobj_ids))

        # context embeddings
        predicate_embedding_context = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb_context, self.predicate_dobj_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.dobj_emb_context, self.predicate_dobj_ids))
        argument_embedding_context = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb_context, self.argument_dobj_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.dobj_emb_context, self.argument_dobj_ids))
        argument_prime_embedding_context = (1 - self.weight) * tf.nn.embedding_lookup(self.center_emb_context,
                                                          self.argument_prime_dobj_ids) + self.weight * self.restrict(
            tf.nn.embedding_lookup(
                self.dobj_emb_context, self.argument_prime_dobj_ids))

        self.overall_predicate_dobj = predicate_embedding

        self.dobj_embedding_handler = predicate_embedding
        self.dobj_embedding_context_handler = argument_embedding_context

        

        self.pred_dobj = tf.reduce_sum(tf.multiply(predicate_embedding, argument_embedding_context), 1) + \
        tf.reduce_sum(tf.multiply(predicate_embedding_context, argument_embedding), 1)

        self.pred_dobj_prime = tf.reduce_sum(tf.multiply(predicate_embedding, argument_prime_embedding_context), 1) + \
        tf.reduce_sum(tf.multiply(predicate_embedding_context, argument_prime_embedding), 1)

    def add_loss(self):
        num_examples = self.get_tensor_shape(self.pred_dobj, 0)

        self.prediction = tf.reshape(tf.concat([self.pred_amod, self.pred_nsubj, self.pred_dobj], 0), [3*num_examples, 1] )
        self.prediction_prime = tf.reshape(tf.concat([self.pred_amod_prime, self.pred_nsubj_prime, self.pred_dobj_prime], 0), [3*num_examples, 1])
           

        all_scores = tf.concat([self.prediction, self.prediction_prime], 1)  # [3 * num_examples, 2]
        all_labels = tf.concat([tf.ones([3 * num_examples, 1]), tf.zeros([3 * num_examples, 1], 1)], 1) 
        self.loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits = all_scores, labels = all_labels))
    def add_optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        self.optimize = optimizer.minimize(self.loss)



# def ws_test(sess, model, data, simlex_corps, num_cross, relation_length, restriction):

#     word2id = data.word2id

#     corp_names = ["v", "n", "adj", "all"]
#     scores = {}

#     with open("ws_gensim_result_c_%d_l_%d_r_%.3f.p "%(int(num_cross), relation_length, restriction), "wb") as fout:
       

#         for i in range(len(simlex_corps)):
#             simlex_corpora = simlex_corps[i]


#             id_score_triplets = []
#             gold_scores = []
#             for words_score_trplet in simlex_corpora:
#                 try:
#                     context1 = word2id[words_score_trplet[0]]
#                     context2 = word2id[words_score_trplet[1]]
#                     gold_score = words_score_trplet[2]
#                     gold_scores.append(gold_score)

#                     id_score_triplets.append([context1, context2])
#                 except:
#                     pass
#             id_score_triplets = np.array(id_score_triplets)

#             feed_dict = {
#                 model.predicate_amod_ids: id_score_triplets[:, 0],
#                 model.argument_amod_ids: id_score_triplets[:, 0],
#                 model.predicate_dobj_ids: id_score_triplets[:, 0],
#                 model.argument_dobj_ids: id_score_triplets[:, 0],
#                 model.predicate_nsubj_ids: id_score_triplets[:, 0],
#                 model.argument_nsubj_ids: id_score_triplets[:, 0]
#             }


            

#             embeddings1 = {"base": {},
#                            "amod": {},
#                            "nsubj": {},
#                            "dobj": {}
#                           }


#             result1 = sess.run([model.base_embedding_handler,
#                                     model.base_embedding_context_handler,
#                                     model.amod_embedding_handler,
#                                     model.amod_embedding_context_handler,
#                                     model.nsubj_embedding_handler,
#                                     model.nsubj_embedding_context_handler,
#                                     model.dobj_embedding_handler,
#                                     model.dobj_embedding_context_handler
#                                     ], feed_dict = feed_dict)

#             embeddings1["base"]["h"],  embeddings1["base"]["t"], embeddings1["amod"]["h"],  embeddings1["amod"]["t"], \
#             embeddings1["nsubj"]["h"],  embeddings1["nsubj"]["t"], embeddings1["dobj"]["h"],  embeddings1["dobj"]["t"] = result1

#             feed_dict = {
#                 model.predicate_amod_ids: id_score_triplets[:, 1],
#                 model.argument_amod_ids: id_score_triplets[:, 1],
#                 model.predicate_dobj_ids: id_score_triplets[:, 1],
#                 model.argument_dobj_ids: id_score_triplets[:, 1],
#                 model.predicate_nsubj_ids: id_score_triplets[:, 1],
#                 model.argument_nsubj_ids: id_score_triplets[:, 1]
#             }

#             embeddings2 = {"base": {},
#                            "amod": {},
#                            "nsubj": {},
#                            "dobj": {}
#                           }

#             result2 = sess.run([model.base_embedding_handler,
#                                     model.base_embedding_context_handler,
#                                     model.amod_embedding_handler,
#                                     model.amod_embedding_context_handler,
#                                     model.nsubj_embedding_handler,
#                                     model.nsubj_embedding_context_handler,
#                                     model.dobj_embedding_handler,
#                                     model.dobj_embedding_context_handler
#                                     ], feed_dict = feed_dict)

#             embeddings2["base"]["h"],  embeddings2["base"]["t"], embeddings2["amod"]["h"],  embeddings2["amod"]["t"], \
#             embeddings2["nsubj"]["h"],  embeddings2["nsubj"]["t"], embeddings2["dobj"]["h"],  embeddings2["dobj"]["t"] = result2

#             for key in embeddings1.keys():
#                 embeddings1[key]["h+t"] = embeddings1[key]["h"] + embeddings1[key]["t"]
#                 embeddings1[key]["[h,t]"] = np.concatenate((embeddings1[key]["h"],  embeddings1[key]["t"]), axis = 1) 

#             for key in embeddings2.keys():
#                 embeddings2[key]["h+t"] = embeddings2[key]["h"] + embeddings2[key]["t"]
#                 embeddings2[key]["[h,t]"] = np.concatenate((embeddings2[key]["h"],  embeddings2[key]["t"]), axis = 1) 


#             relation_result = {}

#             for relation_name in embeddings1.keys():
#                 types_result = {}
#                 for type_name in embeddings1[relation_name].keys():

#                     model_predition_score = [ 1 - cosine(embeddings1[relation_name][type_name][j], embeddings2[relation_name][type_name][j]) for j in range(embeddings1[relation_name][type_name].shape[0])]

            

#                     r, p = spearmanr(gold_scores, model_predition_score)
#                     types_result[type_name] = r
                
                
#                 relation_result[relation_name] = types_result
#                 print(corp_names[i], relation_name, types_result)
#             scores[corp_names[i]] = relation_result
#         pickle.dump(scores, fout)
       
#     return scores
        



def sd_test(sess, model, data, batch_size, c, l, r):
    batch = data.get_sd_test_batch(batch_size)


    feed_dict = {
                    model.predicate_amod_ids: batch["amod"][:, 0],
                    model.argument_amod_ids: batch["amod"][:, 1],
                    model.argument_prime_amod_ids: batch["amod"][:, 2],

                    model.predicate_nsubj_ids: batch["nsubj"][:, 0],
                    model.argument_nsubj_ids: batch["nsubj"][:, 1],
                    model.argument_prime_nsubj_ids: batch["nsubj"][:, 2],
                    
                    model.predicate_dobj_ids: batch["dobj"][:, 0],
                    model.argument_dobj_ids: batch["dobj"][:, 1],
                    model.argument_prime_dobj_ids: batch["dobj"][:, 2],
                }

    pos_amod, pos_nsubj, pos_dobj, neg_amod, neg_nsubj, neg_dobj= sess.run([model.pred_amod, model.pred_nsubj, model.pred_dobj,
        model.pred_amod_prime, model.pred_nsubj_prime, model.pred_dobj_prime], feed_dict=feed_dict)


    
    # with open("pd_gensim_result_c_%d_l_%d_r_%.3f.p "%(int(c), l, r), "wb") as fout:

    amod_acc = np.sum(pos_amod > neg_amod)/ batch_size 
    nsubj_acc = np.sum(pos_nsubj > neg_nsubj) / batch_size
    dobj_acc = np.sum(pos_dobj > neg_dobj) / batch_size
    total_acc = (np.sum(pos_amod > neg_amod) + np.sum(pos_nsubj > neg_nsubj) + np.sum(pos_dobj > neg_dobj))/ (batch_size * 3)

        # pickle.dump({"amod": amod_acc, "nsubj": nsubj_acc, "dobj": dobj_acc, "overall": total_acc}, fout)
    print("sd scores:", amod_acc, nsubj_acc, dobj_acc, total_acc)



    return amod_acc, nsubj_acc, dobj_acc, total_acc


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrain", action="store_true")
    parser.add_argument("-t", "--center_trainable", action="store_true")
    parser.add_argument("-w", "--weight", type=float, default="0.5")
    parser.add_argument("-r", "--restrict", type=float, default="1")
    parser.add_argument("-c", "--cross_validate", type=int, default=0)
    parser.add_argument("-g", "--GPU", type=str, default="0")
    parser.add_argument("-e", "--num_epoch", type=int, default=20)
    parser.add_argument("-l", "--relational_embedding_size", type=int, default=10)
    args = parser.parse_args()
    print(args)
    

    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU
    statsPath = "/home/jbai/ChineseParserUsage/corpus_stats.pkl"
    dataPath = "/home/data/corpora/wikipedia/ParsedChineseWiki/"


    num_cross = str(args.cross_validate)

    train_data = DataLoader(dataPath)
    train_data.read_stats(statsPath)
    train_data.load_pairs_counting("/home/jbai/ChineseParserUsage/count.pkl")
    train_data.load_argument_sample_table("/home/jbai/ChineseParserUsage/sample_table.pkl")

    test_data = DataLoader(dataPath)
    test_data.read_stats(statsPath)
    test_data.load_pairs_counting("/home/jbai/ChineseParserUsage/count.pkl")
    test_data.load_argument_sample_table("/home/jbai/ChineseParserUsage/sample_table.pkl")



    modelPath = "/home/data/corpora/MultiRelChineseEmbeddings/chinese_multirelation.model"
    word2vecModel= Word2Vec.load(modelPath)

    context_wv = KeyedVectors(vector_size=300)
    context_wv.vocab = word2vecModel.wv.vocab
    context_wv.index2word = word2vecModel.wv.index2word  
    context_wv.syn0 = word2vecModel.syn1neg 
            


    pretrain_center_emb = list()
    pretrain_context_emb = list()

    counter = 0

    for i in range(len(train_data.id2word)):
        tmp_w = train_data.id2word[i]
        if tmp_w in context_wv.vocab:
            pretrain_center_emb.append(word2vecModel[tmp_w])
            pretrain_context_emb.append(context_wv[tmp_w])
        else:
            pretrain_center_emb.append(np.zeros(300))
            pretrain_context_emb.append(np.zeros(300))
            counter += 1

    print("empty count", counter)


    pretrain_center_emb = np.asarray(pretrain_center_emb)
    # print("finish load pre-trained vectors")



    m = Model(train_data, args)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print("Model Initialized")

        # trainable set to false
        
        if args.pretrain:

            # Initial assign
            sess.run(m.emb_init, feed_dict={m.emb_placeholder: pretrain_center_emb})
            sess.run(m.emb_init_context , feed_dict={m.emb_placeholder_context: pretrain_context_emb})   
            # print("Center Embeddings Initiated")

        sd_test(sess, m, test_data, 100000, num_cross, args.relational_embedding_size, args.restrict)

        num_epoch = args.num_epoch
        num_batch = 512

        batch_size = 1024
        for epoch in range(num_epoch):
            
            print(" epoch:", str(epoch + 1), "/", num_epoch)
            
            process_bar = tqdm(range(num_batch))
            for i in process_bar:
                batch = train_data.get_sd_train_batch(batch_size)

                feed_dict = {
                    m.predicate_amod_ids: batch["amod"][:, 0],
                    m.argument_amod_ids: batch["amod"][:, 1],
                    m.argument_prime_amod_ids: batch["amod"][:, 2],

                    m.predicate_nsubj_ids: batch["nsubj"][:, 0],
                    m.argument_nsubj_ids: batch["nsubj"][:, 1],
                    m.argument_prime_nsubj_ids: batch["nsubj"][:, 2],
                    
                    m.predicate_dobj_ids: batch["dobj"][:, 0],
                    m.argument_dobj_ids: batch["dobj"][:, 1],
                    m.argument_prime_dobj_ids: batch["dobj"][:, 2],
                }

                loss, _ = sess.run([m.loss, m.optimize], feed_dict=feed_dict)

                process_bar.set_description("Loss: %0.4f" % loss)

            

            # PD test
            amod_acc, nsubj_acc, dobj_acc, total_acc = sd_test(sess, m, train_data, 100000, num_cross, args.relational_embedding_size, args.restrict)
            amod_acc, nsubj_acc, dobj_acc, total_acc = sd_test(sess, m, test_data, 100000, num_cross, args.relational_embedding_size, args.restrict)
           


if __name__ == "__main__":
    main()    