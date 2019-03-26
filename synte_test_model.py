#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

data = pd.read_csv("text.csv", encoding="latin1")
data = data.fillna(method="ffill")
# data.tail(10)
import json
import flask
#from flask import json
import pickle
from flask import request
app = flask.Flask(__name__)
app.config["DEBUG"] = True
# In[2]:
words = list(set(data["words"].values))
n_words = len(words)
n_words
word2idx = {w: i for i, w in enumerate(words)}
idx2word = {i: w for w, i in word2idx.items()}
tags = list(set(data["tag"].values))
n_tags = len(tags)
n_tags
max_len = 60
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: w for w, i in tag2idx.items()}
# In[9]:
padd = list(data['words'])
def to_matrix(padd, n):
    return [padd[i:i+n] for i in range(0, len(padd), n)]
padd_to_2d = list(to_matrix(padd,40))
# print(len(padd_to_2d))
# print(padd_to_2d[0:30])
# In[11]:
new_X = []
for seq in padd_to_2d:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("__PAD__")
    new_X.append(new_seq)
padd_to_2d = new_X
# @app.route('/',methods=['POST','GET'])
# def padd():
#     print(padd_to_2d)
#     return 'jpt'
# app.run()


# In[12]:


# print(len(padd_to_2d))


# # In[13]:


# print(padd_to_2d[:20])


# # In[14]:


padd_y = list(data['tag'])


# # In[15]:


def to_matrix_tag(padd, n):
    return [padd[i:i+n] for i in range(0, len(padd), n)]
padd_to_2d_tag = list(to_matrix_tag(padd_y,40))

# # print(padd_to_2d_tag)


# # In[16]:


y = [[tag2idx[w] for w in s] for s in padd_to_2d_tag]
# # print(y)


# # In[17]:


from keras.preprocessing.sequence import pad_sequences
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])


# # In[18]:


from sklearn.model_selection import train_test_split


# # In[19]:


X_tr, X_te, y_tr, y_te = train_test_split(padd_to_2d, y, test_size=0.1, random_state=2018)


# # In[20]:


batch_size = 32


# # In[21]:


# print(len(X_tr),len(y_tr))
# print(len(X_te),len(y_te))


# # In[22]:


import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K


# # In[23]:


sess = tf.Session()
K.set_session(sess)


# # In[24]:


elmo_model = hub.Module("elmo_hub", trainable=False)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


# # In[25]:


def ElmoEmbedding(x):
    return elmo_model(inputs={
                             "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(batch_size*[max_len])
                       },
                      signature="tokens",
                      as_dict=True)["elmo"]


# # In[26]:


from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda


# # In[27]:


X_tr, X_val = X_tr[:10*batch_size], X_tr[-2*batch_size:]
y_tr, y_val = y_tr[:10*batch_size], y_tr[-2*batch_size:]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)


# # In[28]:


# print(len(X_tr),len(y_tr))
# print(len(X_val),len(y_val))


# # In[29]:


# print(len(X_tr),len(y_tr))


# # In[30]:


input_text = Input(shape=(max_len,), dtype="string")
embedding = Lambda(ElmoEmbedding, output_shape=(None, 1024))(input_text)
x = Bidirectional(LSTM(units=512, return_sequences=True,
                        recurrent_dropout=0.2, dropout=0.2))(embedding)
x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                            recurrent_dropout=0.2, dropout=0.2))(x)
x = add([x, x_rnn])  # residual connection to the first biLSTM
out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)


# # In[31]:


model = Model(input_text, out)


# # In[32]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')


# # In[34]:


checkpoint_path = "training_3/cp-15.ckpt"


# # In[35]:


model.load_weights(checkpoint_path)


# # In[36]:


end_1 = ['vlad', 'silverman', 'resume', 'page', '2', '    ', 'palo', 'alto', 'ca', '94306', '    ', 'vlad', ' ', 'silverman', '   ', 'phone', '   ', '650', '8143773', 'vsilvermangmailcom', '   ', 's', 'u', 'm', 'm', 'a', 'r', 'y', '    ', 'senior', 'software', 'engineering', 'professional', 'with', 'proven', 'record', 'of', 'accomplishments', 'while', 'delivering', 'optimized', 'high', 'quality', 'software', 'solutions', 'on', 'time', 'and', 'within', 'budget', ' ', 'specialties', 'include', '  ', 'bpm', ' ', 'brms', ' ', 'software', 'frameworks', ' ', 'web', 'analytics', ' ', 'business', 'intelligence', ' ', 'data', 'integration', ' ', 'product', 'requirements', '    ', 'tools', ' ', 'env', 'tomcat', 'websphere', 'webmethods', 'weblogic', 'jboss', 'oracle', 'mysql', 'jenkins', 'java', 'bash', 'xml', 'soap', 'rest', 'junit', 'ibm', 'rational', 'tools', 'maven', 'jira', 'confluence', 'puppet', 'labs', 'perforce', 'git', 'ibm', 'jazzbluemix', ' ', 'platformsclouds', 'linux', 'microsoft', 'vmware', 'docker', 'gcp', 'ibm', 'openstack', 'aws', 'oracle', '   ', 'methodologies', '     ', 'rup', 'scrum', 'agile', 'programming', 'open', 'source', 'repositories', 'and', 'metadata', 'cloud', 'computing', 'soa', 'saas', 'paas', 'iaas', 'sdlc', 'bpmn', 'uml', '    ', 'e', 'x', 'p', 'e', 'r', 'i', 'e', 'n', 'c', 'e', '       ', 'nvsoft', 'corp', 'palo', 'alto', 'california', '  ', 'sr', 'integration', 'engineer', 'middleware', 'specialist', ' ', '10201492017', '    ', 'implemented', 'tested', 'and', 'monitored', 'microservices', 'in', 'the', 'datacenter', 'cloud', 'environment', 'for', 'ciscojasper', 'iot', 'platform', 'performing', 'continuous', 'integration', 'and', 'delivery', 'of', 'new', 'microservices', 'ondemand', 'trouble', 'shooting', 'of', 'largescale', 'deployment', 'issues', 'on', 'linux', 'systems', 'started', 'and', 'maintained', 'howto', 'series', 'of', 'knowledge', 'items', 'sharing', 'acquired', 'information', 'about', 'installation', 'integration', 'and', 'deployment', 'for', 'middleware', 'services', 'on', 'privately', 'hosted', 'and', 'public', 'clouds', 'including', 'aws', 'google', 'and', 'ibm', 'clouds', '   ', 'provided', 'configuration', 'maintenance', 'and', 'testing', 'of', 'jibe', 'pipeline', 'framework', 'for', 'apple', 'corp', 'allowing', 'migration', 'of', 'data', 'between', 'heterogeneous', 'systems', 'and', 'services', ' ', 'worked', 'on', 'creating', 'maven', 'based', 'build', 'environment', 'testing', 'import', 'and', 'export', 'components', 'of', 'the', 'jibe', 'framework', 'integration', 'with', 'kafka', 'services', 'monitoring', 'data', 'synchronization', 'between', 'oracle', 'and', 'mongo', 'databases', ' ', 'enabled', 'continuous', 'build', 'and', 'deployment', 'automation', 'for', 'hybrid', 'cloud', 'environment', 'expanding', 'integration', 'coverage', 'for', 'software', 'defined', 'enterprise', 'infrastructure', '    ', 'developed', 'and', 'maintained', 'a', 'toolchain', 'framework', 'for', 'configuring', 'integrating', 'and', 'testing', 'a', 'set', 'of', 'middleware', 'tools', 'used', 'to', 'build', 'corporate', 'vmware', 'products', 'and', 'services', 'project', 'resulted', 'in', 'continuous', 'management', 'of', 'the', 'private', 'cloud', 'based', 'distributed', 'repositories', 'allowing', 'automated', 'test', 'driven', 'synchronization', 'of', 'the', 'toolchain', 'content', ' ', 'created', 'and', 'managed', 'a', 'virtual', 'test', 'lab', 'environment', 'for', 'testing', 'enterprise', 'services', 'inside', 'multitenant', 'cloud', 'infrastructure', '   ', 'designed', 'and', 'implemented', 'adaptive', 'remote', 'testing', 'framework', 'for', 'installation', 'and', 'customization', 'of', 'multitenant', 'cloud', 'environments', 'their', 'integration', 'with', 'distributed', 'data', 'sources', '\xa0\xa0      ', 'ntt', 'data', 'palo', 'alto', 'california', '  ', 'middleware', 'specialist', ' ', '12201292014', '     ', 'responsible', 'for', 'setting', 'up', 'virtual', 'global', 'lab', 'and', 'testing', 'enterprise', 'wide', 'web', 'services', ' ', 'created', 'ats', 'framework', 'and', 'tools', 'used', 'for', 'identity', 'management', 'of', 'users', 'applications', 'and', 'devices', 'for', 'integration', 'of', 'middleware', 'systems', 'with', 'legacy', 'applications', 'had', 'advocated', 'and', 'supported', 'the', 'use', 'of', 'ibm', 'rational', 'tools', 'for', 'automating', 'clm', 'and', 'deployment', 'processes', 'in', 'the', 'cloud', 'environment', 'was', 'utilizing', 'rational', 'approach', 'in', 'dynamic', 'authoring', 'of', 'business', 'rules', 'and', 'performance', 'testing', 'of', 'middleware', 'services', '     ', 'developed', 'optimized', 'and', 'maintained', 'bpm', 'workflows', 'performed', 'continuous', 'discovering', 'authoring', 'and', 'automation', 'of', 'backend', 'business', 'rules', 'for', 'performance', 'tuning', 'provided', 'integration', 'with', 'internal', 'tools', 'through', 'web', 'services', 'and', 'rest', 'api', '          ', 'cisco', 'san', 'jose', 'california', '  ', 'sr', 'integration', 'engineer', 'middleware', 'specialist', ' ', '102011112012', '     ', 'provided', 'automated', 'tests', 'of', 'cloud', 'functionality', 'for', 'cisco', 'middleware', 'communications', 'platform', ' ', 'developed', 'and', 'tested', 'recursive', 'installation', 'utility', 'to', 'perform', 'setup', 'and', 'configuration', 'of', 'the', 'product', 'inside', 'public', 'and', 'private', 'clouds', 'and', 'in', 'sddc', 'cluster', 'environment', '  ', 'implemented', 'data', 'integration', 'with', 'other', 'workflow', 'and', 'rule', 'engines', 'by', 'extending', 'xml', 'based', 'adapters', '    ', 'sony', 'electronics', 'inc', 'san', 'jose', 'california', '  ', 'sr', 'software', 'engineer', 'in', 'test', 'technology', 'consultant', ' ', '9201092011', '     ', 'consulted', 'on', 'business', 'process', 'management', 'solutions', 'worked', 'with', 'webmethods', 'integration', 'server', 'and', 'aris', 'mashzone', 'system', 'developed', 'customer', 's', 'workflows', 'worked', 'with', 'engineering', 'teams', 'on', 'automation', 'and', 'simulation', 'of', 'workflows', 'in', 'crossplatform', 'environment', '       ', 'fico', 'decision', 'management', 'tools', 'san', 'jose', 'california', '  ', 'architect', ' ', '7200792010', '     ', 'designed', 'new', 'functionality', 'for', 'blaze', 'advisor', 'business', 'rules', 'management', 'system', 'maintained', 'the', 'product', 'as', 'market', 'leader', 'in', 'brms', 'industry', ' ', 'product', 'line', 'included', 'desktop', 'ide', 'runtime', 'engine', 'and', 'web', 'based', 'application', ' ', 'platforms', 'include', 'java', 'net', 'and', 'mainframe', 'environment', '     ', 'responsible', 'for', 'integration', 'analyzing', 'and', 'setting', 'priorities', 'to', 'dynamically', 'changing', 'product', 'requirements', 'their', 'maintenance', 'and', 'communication', 'to', 'involved', 'internal', 'teams', ' ', 'provided', 'competitive', 'market', 'analysis', 'help', 'executives', 'defining', 'product', 'strategy', 'and', 'longterm', 'roadmap', '     ', 'initiated', 'implementation', 'of', 'new', 'product', 'features', 'including', 'userfriendly', 'semantic', 'queries', 'interactive', 'refactoring', 'of', 'global', 'rule', 'projects', 'and', 'rule', 'repositories', 'scalable', 'database', 'connectivity', 'and', 'customizable', 'integration', 'with', 'predictive', 'analytics', 'modeling', 'tools', '       ', 'sva', 'corp', 'palo', 'alto', 'california', '    ', 'sr', 'technology', 'specialist', ' ', '9200072007', '      ', 'consulting', 'for', 'sun', 'microsystems', 'executive', 'briefing', 'center', 'in', 'menlo', 'park', ' ', 'product', 'monitors', 'web', 'transactions', 'measures', 'their', 'quality', 'and', 'identifies', 'defects', 'allowing', 'administrators', 'to', 'have', 'dashboard', 'views', 'and', 'multiple', 'graphical', 'reports', 'of', 'datacenter', 'performance', ' ', 'developed', 'customizable', 'webbased', 'reporting', 'infrastructure', 'optimized', 'backend', 'business', 'rules', 'repository', 'improved', 'reliability', 'of', 'nightly', 'builds', '     ', 'consulting', 'for', 'ibm', 'db2', 'management', 'group', 'ibm', 'silicon', 'valley', 'lab', 'on', 'presales', 'demo', 'of', 'ibm', 'middleware', 'and', 'database', 'technologies', 'for', 'insurance', 'industry', ' ', 'under', 'tight', 'schedule', 'developed', 'functional', 'websphere', 'portal', 'providing', 'realtime', 'workflow', 'and', 'communication', 'with', 'distributed', 'dbms', 'db2', 'oracle', 'integrated', 'crossplatform', 'search', 'and', 'content', 'management', 'portlets', ' ', 'developed', 'collaborative', 'portlets', 'configuration', 'of', 'ws', 'portal', 'for', 'customer', 'needs', '     ', 'consulting', 'through', 'accenture', 'on', 'jrules', 'product', 'business', 'rules', 'management', 'system', 'from', 'ilog', ' ', 'integration', 'of', 'jrules', 'with', 'middleware', 'frameworks', 'application', 'servers', 'and', 'legacy', 'systems', ' ', 'improvement', 'of', 'dynamic', 'binding', 'functionality', 'by', 'using', 'internal', 'xml', 'schema', 'attributes', 'for', 'rapid', 'and', 'customized', 'generation', 'of', 'business', 'object', 'model', ' ', 'deployment', 'of', 'production', 'rules', 'and', 'correspondent', 'object', 'model', 'as', 'a', 'web', 'service', 'their', 'bpm', 'integration', '       ', 'education', '    ', 'ms', 'bs', 'in', 'computer', 'science', ' ', 'moscow', 'state', 'technical', 'university', ' ', 'mirea', '      ', 'certificates', 'in', 'java', 'oopood', 'brms', 'and', 'bpm', 'development', 'webmethods', '                          ', 'ibm', 'rational', 'team', 'concert', 'rtc', 'ciscojasper', 'cc', 'operations', ' ', 'courses', 'at', 'universities', 'of', 'california', 'uc', 'system', 'oracle', 'ibm', 'fico', 'sony', 'cisco', '      ', 'awards', ' ', 'smithsonian', 'award', 'for', 'innovative', 'internet', 'services', 'att', 'labs']

end_2 = ['AWS', 'Architect', '/', 'Cloud', 'Engineer', '/', 'Data', 'Engineer', 'Specialty', ':', 'Software', 'Engineering', '–', 'Systems', 'Architecture', '–', 'Programming', '–', 'Analytics', '–', 'Cloud', 'Engineering', 'Vijay', 'Ram', 'Veeraswamy', 'E', 'mail:ram_vijay@hotmail.com', 'Mobile', ':', '484', '273', '5430', 'Visa', 'Status', ':', 'H1B', 'www.linkedin.com/pub/vijay', 'ram', 'veeraswamy/1a/247/951/', 'SUMMARY', 'Data', 'Engineer', '/', 'Cloud', 'Engineer', 'with', '12', '+', 'years', 'of', 'experience', 'in', 'Application', 'Analysis', ',', 'Infrastructure', 'Design', ',', 'Development', ',', 'Integration', ',', 'deployment', ',', 'and', 'Maintenance', '/', 'Support', 'for', 'AWS', 'cloud', 'Computing', ',', 'Enterprise', 'Search', 'Technologies', ',', 'Artificial', 'Intelligence', ',', 'Micro', 'services', ',', 'Web', ',', 'Enterprise', 'based', 'Software', 'Applications', '.', 'Hands', 'on', 'experience', 'with', 'ELK', '(', 'Elasticsearch', ',', 'Logstash', 'and', 'Kibana', ')', ',', 'working', 'knowledge', 'on', 'Kafka', 'and', 'other', 'stream', 'infrastructure', '.', 'Wide', 'experience', 'on', 'Data', 'Mining', ',', 'Real', 'time', 'Analytics', ',', 'Business', 'Intelligence', ',', 'Machine', 'Learning', 'and', 'Web', 'Development', '.', '\xa0 ', 'Exposure', 'on', 'usage', 'of', 'Apache', 'Kafka', 'develop', 'data', 'pipeline', 'of', 'logs', 'as', 'a', 'stream', 'of', 'messages', 'using', 'producers', 'and', 'consumers', '.', '\xa0 ', 'Experience', 'in', 'coordinating', 'Cluster', 'services', 'through', 'ZooKeeper', '.', 'Integrated', 'data', 'from', 'multiple', 'sources', 'like', 'Rally', ',', 'Sonar', ',', 'Git', ',', 'SVN', ',', 'Jenkins', ',', 'Testrail', 'using', 'logstash', '.', '\xa0 ', 'Hands', 'on', 'experience', 'in', 'Creating', 'indexes', 'and', 'mapping', '/', 'schema', 'and', 'loading', 'data', 'in', 'to', 'ElasticSearch(Full', 'load', 'and', 'incremental', ')', ',', 'experience', 'in', 'using', 'RabbitMQ', 'Message', 'bus', 'Experience', 'in', 'Implementing', 'full', 'text', 'search', ',', 'Synonym', ',', 'Filters', 'in', 'Elasticsearch', 'engine', ',', 'implemented', 'administered', 'and', 'maintained', 'ElasticSearch', 'Machine', 'Learning', 'Algorithms', ':', 'supervised', 'learning', '(', 'linear', '/', 'logistic', 'regression', ',', 'neural', 'network', ',', 'SVM', ',', 'random', 'forest', ')', 'and', 'unsupervised', 'learning', '(', 'PCA', ',', 'K', 'mean', ',', 'anomalous', 'detection', ',', 'EM', ',', 'LSH', ',', 'LDA', ')', '.', 'Define', 'and', 'develop', 'analytical', ',', 'statistical', 'and', 'machine', 'learning', 'algorithms', 'for', 'advanced', 'analysis', 'and', 'prediction', 'Expertise', 'in', 'setting', 'up', 'the', 'enterprise', 'infrastructure', 'on', 'AWS', 'Cloud', 'Administration', 'like', 'EC2', 'Instance', ',', 'ELB', ',', 'EBS', ',', 'S3', 'Bucket', ',', 'Security', 'Groups', ',', 'Auto', 'Scaling', ',', 'AMI', ',', 'RDS', ',', 'Route', '53', ',', 'Cloud', 'Front', ',', 'Cloud', 'Watch', ',', 'Cloud', 'Formation', ',', 'IAM', 'Cloud', 'Formation', '&', 'VPC', 'services', '.', '\xa0 ', 'Re', 'architected', 'and', 'administerd', 'cloud', 'topology', 'restructure', 'as', 'part', 'of', 'drive', 'towards', 'passion', 'of', 'continous', 'innovation', ',', 'this', 'initiative', 'saved', 'the', 'company', 'over', 'a', 'million', 'dollar', 'on', 'our', 'Dev', '/', 'QA', 'cloud', 'for', 'our', 'account', '.', 'Mentored', 'implementation', 'teams', 'and', 'conduct', 'deep', 'dive', 'and', 'hands', 'on', 'education', '/', 'training', 'sessions', 'to', 'transfer', 'knowledge', 'to', 'developers', '/', 'product', 'managers', 'Hands', 'on', 'experience', 'with', 'container', 'virtualization', 'Docker', ',', 'AWS', 'cloud', 'infrastructure', ',', 'including', 'CI', '/', 'CD', 'tools', 'such', 'as', 'Jenkins', ',', 'configuration', 'management', 'with', 'GIT', ',', 'Maven', ',', 'and', 'Chef', '.', 'Experience', 'with', 'Splunk', 'Searching', 'and', 'Reporting', ',', 'Knowledge', 'Objects', ',', 'Administration', ',', 'Add', 'On', "'s", ',', 'Dashboards', ',', 'Clustering', 'and', 'Forwarder', 'Management', 'Excellent', 'background', 'in', 'architecting', 'application', 'using', 'Java', '1.7', ',', '1.8', ',', 'J2EE', '2.x', ',', 'JSP', ',', 'Servlets', ',', 'Spring', 'core', ',', 'Spring', 'DAO', ',', 'Spring', 'Security', ',', 'Spring', 'Annotations', ',', 'Spring', 'cloud', ',', 'Springboot', ',', 'Microservice', ',', 'Eureka', 'server', ',', 'Lambda', ',', 'Spring', 'MVC', ',', 'Java', 'Code', 'Refactoring', ',', 'SOA', 'architecture', ',', 'Spring', 'Web', 'services(SOAP', ')', ',', 'Spring', 'AOP', ',', 'H20.ai', ',', 'Api.ai', ',', 'Mahout', ',', 'Elasticsearch', ',', 'Logstash', ',', 'Kibana', ',', 'Mahout', '.', 'Experience', 'in', 'branching', ',', 'tagging', 'and', 'maintaining', 'the', 'version', 'across', 'the', 'environments', 'working', 'on', 'Software', 'Configuration', 'Management', '(', 'SCM', ')', 'tools', 'like', 'SVN', 'and', 'GIT', '.', '\xa0 ', 'Expertise', 'in', 'analyzing', 'client', 'requirements', ',', 'build', 'cost', 'effective', 'and', 'scalable', 'application', 'architecture', '.', '\xa0 ', 'Managed', 'large', 'amounts', 'of', 'structured', ',', 'semi', 'structured', ',', 'and', 'unstructured', 'data', 'across', 'multiple', 'data', 'Excellent', 'background', 'in', 'deploying', 'applications', 'using', 'on', 'BEA', 'WebLogic', 'Application', 'Server', ',', 'WebSphere', 'Application', 'Server', '(', 'WAS', ',', 'Jetty', ',', 'Apache', 'Tomcat', ',', 'JDBC', 'and', 'various', 'Databases', 'like', 'Oracle', ',', 'Sybase', ',', 'and', 'Microsoft', 'SQL', 'server', ',', 'DB2', ',', 'Microsoft', 'SQL', 'Server', '.', 'Strong', 'experience', 'on', 'various', 'development', 'environments', 'like', 'STS', '(', 'Spring', 'Tool', 'Suite', ')', ',', 'IBM', 'RAD', ',', 'Visio', ',', 'Eclipse', ',', 'Subversion', '(', 'SVN', ')', ',', 'Quartz', '.', 'Technical', 'Skills', ':', 'Operating', 'System', ':', 'Windows', ',', 'UNIX', ',', 'Linux', ',', 'Solaris', ',', 'Development', ',', 'Methodologies', ':', 'Agile', 'Methodology', '(', 'Scrum', 'and', 'Extreme', ',', 'Kanban', ')', ',', 'Waterfall', 'model', 'Frameworks', 'and', 'Architectures', ':', 'AWS', 'cloud', ',', 'Microservices', ',', 'JSON', ',', 'PMD', 'Plugins', ',', 'and', 'AWS', '.', 'Cloud', 'Platform', '\xa0', 'AWS', '\xa0', '(', 'Cloud', 'formation', ',', 'ELB', ',', 'EC2', ',', 'S3', ',', 'Cloud', 'Watch', ',', 'RDS', ',', 'Elastic', 'Cache', ',', 'IAM', ',', 'kinesis', ')', 'SOA', ',', 'SOAP', ',', 'Model', 'View', 'Controller', '(', 'MVC', ')', ',', 'Spring', 'MVC', ',', 'Spring', '2.5', ',', 'Spring', 'Batch', '2.2', ',', 'Spring', 'Integration', ',', '\xa0 ', 'Database', ':', 'MongoDB', ',', 'Oracle', ',', 'DynamoDB', ',', 'MongoDB', ',', 'Arora', ',', 'Sybase', ',', 'DB2', ',', 'MS', 'ACCESS', ',', 'SQL', 'Server', '.', 'Languages/', 'API/', 'Technologies', ':', 'Java', '1.x', ',', 'SQL', ',', 'Servlets', ',', 'JSP', ',', 'JNDI', ',', 'Java', 'Beans', ',', 'Java', 'script', ',', 'Bash', 'script', '.', 'Servers', ':', 'Jetty', 'TOMCAT', '7.0', ',', 'APACHE', 'WEB', 'SERVER', '2.1', ',', 'TC', 'Server', ',', 'WebLogic', ',', 'WebSphere', 'App', 'Server', '9.x', 'Tools', ':', 'SVN', '(', 'Subversion', ')', ',', 'GIT', ',', 'Jenkins', ',', 'Chef', ',', 'Ansible', ',', 'Docker', ',', 'AWS', 'SDK', ',', 'AWS', 'CLI', ',', 'Putty', ',', 'PuttyGen', ',', 'Wireshark', ',', 'Fiddler', ',', 'H2o.ai', ',', 'API.ai', ',', 'Mahout', ',', 'ELK', ',', 'OpenNLP', 'etc', '.', 'IDE', ':', 'Spring', 'Tool', 'Suite', ',', 'RAD', '7.0/7.5', ',', 'Eclipse', ',', 'RoboMongo', ',', 'Toad', ',', 'DB2', 'Client', ',', 'DB', 'Squirrel', ',', 'SOAP', 'UI', 'Certifications', ':', 'Sun', 'Certified', 'Java', 'Programmer', '5', 'Sun', 'certification', 'AWS', 'Certified', 'Developer', '–', 'Associate', '(', 'AWS', 'ADEV', '8026', ')', 'Amazon', 'certification', 'AWS', 'Certified', 'Solutions', 'Architect', '–', 'Associate', '(', 'AWS', 'ASA', '32095', ')', 'Amazon', 'certification', 'AWS', 'Certified', 'SysOps', 'Administrator', '–', 'Associate', '(', 'AWS', 'ASOA', '8047', ')', 'Amazon', 'certification', 'Education', ':', 'Bachelor', 'of', 'Engineering', 'from', 'Madras', 'University', '(', 'India', ')', 'Verizon', 'Wireless', ',', 'Lowell', ',', 'MA', '(', 'Jan', '17', '–', 'Till', 'date', ')', 'MyVerizon', 'Web', 'Portal', 'provides', 'customers', 'with', 'a', 'variety', 'of', 'self', 'serve', 'capabilities', 'involving', 'the', 'management', 'of', 'the', 'wireless', 'account', 'and', 'the', 'ability', 'to', 'view', '/', 'pay', 'their', 'bill', 'via', 'VerizonWireless.com', '.', 'Through', 'the', 'portal', ',', 'customer', 'can', 'perform', 'functionalities', 'such', 'as', 'profile', 'management', ',', 'device', 'management', ',', 'plan', '/', 'feature', 'management', 'and', 'payment', '/', 'wallet', 'management', 'etc', '.', 'This', 'application', 'has', 'three', 'different', 'sub', 'applications', 'run', 'on', 'three', 'different', 'platforms', ',', 'namely', 'legacy', 'on', 'J2EE', '/', 'Struts', '/', 'WL', ',', 'OMNI', 'on', 'ATG', '/', 'Endeca', '/', 'WL', 'and', 'microservices', '/', 'sites', 'on', 'Spring', 'Boot', '/', 'Cloud', '/', 'Tomcat', '.', 'Responsibilities', ':', 'As', 'a', 'AWS', 'Architect', '/', 'Data', 'Engineer', 'my', 'responsibilities', 'in', 'this', 'project', 'are', ':', 'Implemented', 'Elasticsearch', 'with', '15', 'nodes', 'sharding', '.', 'The', 'system', 'was', 'created', 'in', 'AWS', 'cloud', 'as', 'PAAS', '.', 'The', 'development', 'and', 'test', 'of', 'the', '15', 'node', 'done', 'on', 'VirtualBox', 'machines', ',', 'physical', 'machines', 'before', 'deployment', 'to', 'the', 'Cloud', '.', 'Developed', 'crawlers', 'to', 'crawl', 'Verizon', 'internal', 'websites', ',', 'Sharepoint', ',', 'and', 'batch', ',', 'and', 'index', 'them', 'in', 'Elasticsearch', 'using', 'Nutch', '.', 'Composted', 'request', 'to', 'query', 'data', 'on', 'ElsticSearch', 'using', 'http', 'client', '.', 'Develop', 'RESTful', 'APIs', 'to', 'schedule', 'and', 'monitor', 'crawlers', '.', 'Experience', 'in', 'implementing', 'full', 'text', '\xa0', 'search', '\xa0', 'platform', 'using', 'NoSQL', 'Elasticsearch', 'engine', ',', 'allowing', 'for', 'much', 'faster', ',', 'more', 'scalable', 'and', 'more', 'intuitive', 'user', '\xa0', 'searches', '\xa0', 'for', 'our', 'database', '.', '\xa0 ', 'Performed', 'real', 'time', 'analysis', 'of', 'the', 'incoming', 'data', 'using', 'Kafka', 'consumer', 'API', ',', 'Kafka', 'topics', ',', 'Spark', 'Streaming', '.', '\xa0', 'Created', 'template', 'files', 'for', 'logstash', 'in', 'Elasticsearch', 'to', 'handle', 'not', 'analyzed', 'string', 'inputs', '.', 'Architected', 'and', 'implemented', 'advanced', 'Machine', 'Learning', 'system', 'using', 'Mahout', 'Machine', 'learning', 'and', 'recommendation', 'system', 'using', 'both', 'ItemSimilarity', 'and', 'UserNeighborhood', '.', 'Implemented', 'and', 'trained', 'model', 'for', 'Mahout', 'user', 'based', 'recommenders', ',', 'item', 'based', 'recommender', 'then', 'integrated', 'with', 'Elasticsearch', 'to', 'give', 'precise', 'recommendations', 'for', 'the', 'search', 'query', '.', 'Architected', 'entire', 'on', 'premise', 'data', 'center', 'stack', 'to', 'AWS', 'cloud', 'resources', 'using', 'cloud', 'formation', 'template', 'leveraging', 'Ansible', 'playbooks', 'for', 'automated', 'provisioning', ',', 'configuration', 'management', ',', 'and', 'application', 'deployment', '.', 'Collaborated', 'with', 'the', 'product', ',', 'data', ',', 'and', 'infrastructure', 'architects', 'and', 'was', 'responsible', 'for', 'the', 'define', 'design', 'deliver', 'the', 'technical', 'architectures', ',', 'patterns', ',', 'technical', 'quality', ',', 'risks', ',', 'fitness', 'for', 'purpose', 'and', 'operability', 'of', 'technical', 'architecture', 'solutions', '.', 'Extensive', 'implementiation', 'and', 'expertised', 'knowledge', 'on', 'compute', '/', 'storage', '/', 'Databases', 'stacks', 'like', 'EC2', ',', 'S3', ',', 'RDS', ',', 'EFS', ',', 'Glacier', ',', 'Lambda', 'Elastic', 'Beanstalk', ',', 'EBS', ',', 'Lamda', 'functions', ',', 'DynamoDB', ',', 'Kinesis', '.', 'Designed', 'roles', 'and', 'groups', 'for', 'users', 'and', 'resources', 'using', 'AWS', 'Identity', 'Access', 'Management', '(', 'IAM', ')', '.', '\xa0 ', 'Excellent', 'assimilation', 'of', 'Cloudwatch', 'service', 'for', 'monitoring', 'the', 'servers', 'performance', ',', 'CPU', 'Utilization', ',', 'diskusage', ',', 'maintained', 'user', 'accounts', 'IAM', ',', 'RDS', ',', 'Route', '53', 'services', 'in', 'AWS', 'cloud', '.', 'SonarQube', 'was', 'used', 'to', 'find', 'the', 'trends', 'of', 'lagging', 'and', 'leading', 'quality', 'indicators', '.', '\xa0 ', 'Estimated', 'monthly', 'bill', 'and', 'identified', 'areas', 'of', 'development', 'to', 'reduce', 'our', 'monthly', 'costs', 'and', 'even', 'compare', 'it', 'with', 'other', 'service', 'providers', 'using', 'AWS', 'simple', 'calculator', 'Thorough', 'understanding', 'of', 'infrastructure', 'in', 'areas', 'of', 'firewalls', ',', 'security', ',', 'load', 'balancers', ',', 'hypervisors', ',', 'monitoring', ',', 'network', 'topologies', ',', 'storage', ',', 'AWS', 'IAM', ',', 'NACLs', ',', 'Security', 'Groups', ',', 'Bastion', 'Host', ',', 'NAT', 'Excellent', 'understanding', 'of', 'Java', '1.8', 'new', 'features', 'such', 'as', 'Lamda', 'expressions', ',', 'Functional', 'Interfaces', ',', 'forEach', '(', ')', 'Iterable', 'interface', 'Extensively', 'used', 'Splunk', 'log', 'for', 'errors', 'and', 'exceptions', ',', 'business', 'logic', ',', 'logging', 'for', 'debugging', 'for', 'the', 'aws', 'cloud', 'infrastructure', '.', 'Used', 'maven', 'as', 'plugins', 'or', 'goals', 'executables', ',', 'build', 'profiles', ',', 'build', 'tool', ',', 'maintaining', 'artifacts', ',', 'centralized', 'repository', 'and', 'Dependency', 'management', '.', 'Planning', ',', 'deploying', ',', 'monitoring', ',', 'and', 'maintaining', 'Amazon', 'AWS', 'cloud', 'infrastructure', 'consisting', 'of', 'multiple', 'EC2', 'nodes', 'and', 'required', 'in', 'the', 'environment', '.', 'Used', 'security', 'groups', ',', 'network', 'ACLs', ',', 'Internet', 'Gateways', ',', 'NAT', 'instances', 'and', 'Route', 'tables', 'to', 'ensure', 'a', 'secure', 'zone', 'for', 'organizations', 'in', 'AWS', 'public', 'cloud', '.', 'Environment', ':', 'AWS', ',', 'Atlassian', 'Bitbucket', ',', 'maven', ',', 'Sonar', ',', 'Gradle', ',', 'XML', ',', 'JAVA', ',', 'Puppet', ',', 'Docker', ',', 'Open', 'Shift', ',', 'Kubernetes', ',', 'Ansible', ',', 'CodeDeploy', ',', 'Splunk', ',', 'Web', 'Sphere', ',', 'Apache', 'Tomcat', ',', 'JSON', ',', 'JQuery', ',', 'JAVA', ',', 'Python', ',', 'TFS', ',', 'UNIX', '/', 'Linux', ',', 'Windows', 'Server', ',', 'PostgreSQL', ',', 'MS', 'SQL', 'Server', ',', 'ELK', ',', 'H2o.ai', ',', 'Api.ai', ',', 'Mahout', '.', 'Automated', 'Financial', 'Systems', ',', 'Exton', ',', 'PA', '(', 'Jun', '14', '–', 'Dec', '2016', ')', 'AFSVision', 'is', 'a', 'Banking', 'and', 'Leveraging', 'product', 'for', 'commercial', 'banking', ',', 'it', 'is', 'a', 'manufacturing', 'model', 'designed', 'to', 'address', 'lines', 'of', 'business', 'through', 'all', 'stages', 'of', 'work', ',', 'from', 'origination', 'to', 'booking', 'to', 'servicing', 'to', 'reporting', '.', 'AFSVision', 'is', 'an', 'enterprise', 'wide', 'approach', 'to', 'lending', '.', 'It', 'encompasses', 'lending', 'portfolio', ',', 'offering', 'Wholesale', ',', 'Corporate', ',', 'Middle', 'Market', ',', 'and', 'Small', 'Business', '.', 'AFS', 'Vision', 'handles', 'the', 'complete', 'automation', 'of', 'the', 'Information', 'gathering', ',', 'Collateral', 'Management', ',', 'Financial', 'analysis', ',', 'Risk', 'assessment', '.', 'It', 'is', 'a', 'one', 'stop', 'for', 'Commercial', 'banking', 'lending', 'operation', 'portfolio', '.', 'Responsibilities', ':', 'As', 'a', 'AWS', 'Architect', 'my', 'responsibilities', 'in', 'this', 'project', 'are', ':', 'Design', ',', 'architected', 'and', 'elevated', 'Jenkin', 'jobs', 'to', 'run', 'the', 'cloud', 'formation', 'template', 'and', 'ansible', 'playbooks', 'and', 'automated', 'the', 'entire', 'baked', 'AMI', '’s', '.', 'Extensive', 'implementiation', 'and', 'expertised', 'knowledge', 'on', 'compute', '/', 'storage', '/', 'Databases', 'stacks', 'like', 'EC2', ',', 'S3', ',', 'RDS', ',', 'EFS', ',', 'Glacier', ',', 'Lambda', 'Elastic', 'Beanstalk', ',', 'EBS', ',', 'Lamda', 'functions', ',', 'DynamoDB', ',', 'Kinesis', '.', 'Designed', 'roles', 'and', 'groups', 'for', 'users', 'and', 'resources', 'using', 'AWS', 'Identity', 'Access', 'Management', '(', 'IAM', ')', '.', '\xa0 ', 'Analzye', ',', 'identify', 'and', 'resolve', 'security', 'vulnerabilty', 'on', 'the', 'OSI', 'layers', 'and', 'securing', 'the', 'Cloud', 'infrastructure', 'from', 'attacks', '.', 'Extensive', 'working', 'knowledge', 'and', 'understanding', 'of', 'wireshark', 'and', 'Fiddler', 'tools', 'to', 'sniff', 'packets', 'and', 'Fortify', 'our', 'ecosystem', 'based', 'on', 'the', 'Http', 'traffic', 'and', 'patterns', '.', 'Implementation', 'of', 'SSL', 'certificates', ',', 'Security', ',', 'Authentication', ',', 'Authorization', ',', 'Data', 'privacy', ',', 'Identity', 'and', 'Access', 'Management', '(', 'IAM', ')', ',', 'Cryptography', '/', 'Key', 'Management', ',', 'Access', 'Controls', 'and', 'Security', '.', 'Installed', ',', 'configure', ',', 'designed', 'and', 'architected', 'messaging', 'services', 'from', 'scratch', 'using', 'RabbitMQ', 'messaging', 'broker', 'and', 'its', 'clients', '.', 'Designed', 'and', 'Architected', 'Information', 'gathering', 'and', 'the', 'Risk', 'Assement', 'modules', 'from', 'scratch', 'using', 'Spring', 'Core', 'and', 'Spring', 'DAO', ',', 'Spring', 'Data', ',', 'Microservices', 'using', 'spring', 'boot', ',', 'spring', 'cloud', 'and', 'Netflix', 'Eureka', 'Server', ',', 'Eureka', 'server', 'Architected', 'Microservices', 'augmented', 'with', 'Eurekaserver', 'and', 'FienClient', 'to', 'communicate', 'with', 'the', 'third', 'party', 'Message', 'brokers', 'and', 'ILM', 'clients', 'using', 'SQS', ',', 'SNS', 'containerized', 'using', 'Docker', 'Implement', 'OAuth', 'and', 'OspenID', 'Connect', 'flows', 'using', 'JSON', 'Web', 'Tokens', 'to', 'create', 'a', 'distributed', 'authentication', 'mechanism', 'for', 'application', 'layer', '.', 'Extensive', 'and', 'implementation', 'knowledge', 'on', 'Aggregator', ',', 'Chained', ',', 'Asynchronous', 'Messaging', ',', 'Shared', 'Data', 'Microservice', 'Design', 'Pattern', 'to', 'accomplish', 'a', 'robust', 'microservice', 'application', '.', '\xa0 ', 'Designed', 'Rest', 'clients', 'with', 'Fein', ',', 'Client', 'Side', 'Load', 'Balancing', 'with', 'Zuul', ',', 'Ribbon', ',', 'circuit', 'breakers', 'for', 'failing', 'method', 'calls', 'using', 'the', 'Netflix', 'Hystrix', 'for', 'fault', 'tolerance', 'and', 'avoid', 'latency', '.', 'Involved', 'in', 'phase', 'to', 'phase', 'decomposition', 'of', 'Monolith', 'application', ',', 'decomposing', 'the', 'rest', 'services', 'to', 'microservices', 'and', 'migrating', 'the', 'developed', 'archives', 'to', 'AWS', 'cloud', '.', 'Environment', ':', 'Java', '1.7,1.8', ',', 'j2EE', ',', 'SOAP', 'Webservices', ',', 'Spring', 'microservices', ',', 'Jenkins', ',', 'Ansible', ',', 'Chef', ',', 'AWS', 'stack', '(', 'Cloudformation', ',', 'NAT', 'instances', ',', 'SQS', ',', 'EC2', ',', 'ELB', ',', 'EBS', ',', 'VPC', ',', 'Radis', ',', 'RDS', ')', ',', 'Maven', ',', 'Tomcat', 'server', ',', 'Eclipse', ',', 'Oracle', ',', 'DB2', ',', 'WAS', '7.x', ',', 'RAD', ',', 'Amazon', 'web', 'services', ',', 'MongoDB', ',', 'RabbitMQ', ',', 'Micro', 'services', '(', 'Eureka', 'Discovery', 'server', ')', 'Comcast', 'Spotlight', ',', 'West', 'Chester', ',', 'PA', 'IVerify', '/', 'Converged', '/', 'CTC', '(', 'Nov', '13', '–', 'Jun', '14', ')', 'DESCRIPTION', ':', 'IVerify', 'Application', 'Programming', 'Interface', '(', 'API', ')', 'would', 'communicate', 'with', 'all', 'Spotlight', 'Ad', 'Servers', 'and', 'calculate', 'the', 'actual', 'vs.', 'expected', 'online', 'advertising', 'activity', ',', 'regardless', 'of', 'which', 'Ad', 'Server', 'the', 'asset', 'was', 'deployed', 'to', '.', 'A', 'front', 'end', 'user', 'interface', 'will', 'allow', 'both', 'NFC', 'and', 'Local', 'Sales', 'AEs', 'and', 'Sales', 'Coordinators', 'to', 'view', 'the', 'activity', '/', 'performance', 'for', 'each', 'order', 'or', 'campaign', '.', 'iVerify', 'will', 'enable', 'consistent', 'reporting', 'across', 'all', 'online', 'properties', 'and', 'products', 'for', 'both', 'national', 'and', 'local', 'marketing', 'campaigns', '.', 'Responsibilities', ':', 'As', 'a', 'Senior', 'Developer', 'my', 'responsibilities', 'in', 'this', 'project', 'are', ':', 'Designed', 'and', 'developed', 'backend', 'layer(Service', 'layer', ')', 'using', 'Spring', 'DAO', ',', 'Spring', 'Data', 'for', 'MongoDB', ',', 'Spring', 'Restful', 'webservice', '.', 'Designed', 'and', 'developed', 'web', 'tier', 'components', 'for', 'the', 'reporting', 'modules', 'using', 'AngularJS', ',', 'Spring', 'MVC', ',', 'JSP', ',', 'and', 'Servlets', 'also', 'enhanced', 'some', 'of', 'the', 'functionalities', 'in', 'iVerify', 'reporting', 'web', 'screens', '.', 'Developed', 'Unit', 'Test', 'cases', 'using', 'JUNIT', ',', 'Spring', 'JUNIT', 'annotation', ',', 'Mockito', ',', 'PowerMockito', 'testing', 'to', 'statisfy', 'block', 'coverage', 'Maven', 'Emma', 'and', 'Checkstyle', 'goals', '.', 'Developed', 'Consumers', '(', 'Webservice', 'client', ')', 'and', 'Providers', '.', 'Developed', 'consumer', 'webservice', 'client', 'to', 'intract', 'with', 'various', 'ad', 'servers', 'namely', ',', 'OAS', ',', 'Adjuster', ',', 'DART', 'etc', '.', 'Environment', ':', 'Java', '6', ',', 'j2EE', ',', 'SOAP', 'Webservices', ',', 'Spring', 'security', ',', 'Spring', 'Batch', ',', 'Spring', 'core', ',', 'Spring', 'DAO', ',', 'Spring', 'JMS', ',', 'Apache', 'CXF', ',', 'Maven', ',', 'Tomcat', 'server', ',', 'Eclipse', ',', 'Continuum', ',', 'JUnits', 'and', 'Mockito', ',', 'Postgres', ',', 'Microsoft', 'SQL', 'Server', ',', 'Oracle', '.', 'Statefarm', 'Insurance', ',', 'Bloomington', ',', 'IL', 'Staff', 'Aug', ',', 'Bloomington', ',', 'USA', 'ESignature', '(', 'Nov', '12', '–', 'Nov', '13', ')', 'DESCRIPTION', ':', 'One', 'component', 'of', 'State', 'Farm', 'Customer', 'Driven', 'Evolution', '(', 'CDE', ')', '\xa0', 'is', 'the', 'development', 'of', 'a', 'simple', ',', 'seamless', ',', 'integrated', 'customer', 'platform', '.', 'This', 'integrated', 'platform', 'will', 'provide', 'customers', 'with', 'the', 'ability', 'to', 'acquire', 'and', 'service', 'all', 'of', 'their', 'State', 'Farm', '®', 'insurance', 'and', 'financial', 'services', 'products', 'across', 'all', 'access', 'points', '.', 'To', 'accomplish', 'these', 'objective', ',', 'customers', 'will', 'have', 'the', 'capability', 'to', 'sign', 'an', 'agreement', 'with', 'State', 'Farm', '(', 'such', 'as', 'purchasing', 'a', 'policy', ')', 'over', 'any', 'approved', 'access', 'channel', 'if', 'there', 'is', 'a', 'high', 'level', 'of', 'assurance', 'of', 'the', 'person', '’s', 'identity', 'and', 'there', 'are', 'no', 'restrictions', 'on', 'the', 'type', 'of', 'signature', 'for', 'that', 'document', '.', '\xa0 ', 'Objective', 'of', 'the', 'project', 'is', 'to', 'provide', 'online', 'electronic', 'signature', 'capabilities', '(', 'Online', ',', 'Agent', 'Office', ')', 'to', 'various', 'lines', 'of', 'businesses', 'at', 'State', 'Farm', '.', 'The', 'online', 'eSign', 'capabilities', 'are', 'achieved', 'by', 'interacting', 'with', 'DocuSign', 'API', '’s', '.', 'Responsibilities', ':', 'As', 'a', 'Lead', 'Developer/', 'Tech', 'Lead', 'my', 'responsibilities', 'in', 'this', 'project', 'are', ':', 'Owned', 'the', 'complete', 'module', 'of', 'the', 'project', 'under', 'development', ',', 'Mentored', 'junior', 'team', 'members', 'for', 'this', 'project', 'Coding', 'on', 'various', 'farmework', 'which', 'includes', 'Spring', 'Core', ',', 'Spring', 'JDBC', ',', 'Spring', 'Webservices', ',', 'Spring', 'AOP', ',', 'Spring', 'Security', ',', 'Apache', 'CXF', ',', 'Maven', ',', 'TC', 'server', ',', 'Mule', 'Server', ',', 'WSRR', ',', 'Jenkins', ',', 'Postgres', '.', 'Extensively', 'used', 'Spring', 'Annotations', 'Code', 'Refactoring', 'as', 'per', 'NGSA', '(', 'Next', 'Generation', 'Software', 'Architecture', ')', 'standard', ',', 'This', 'includes', 'using', 'STS', 'as', 'the', 'development', 'IDE', ',', 'Maven', 'Toolset', ',', 'Sonar', ',', 'EclEmma', 'plugin', '.', 'Environment', ':', 'Java', '6', ',', 'Spring', 'Webservices', ',', 'WSRR', ',', 'Spring', 'core', ',', 'Spring', 'DAO', ',', 'Spring', 'JMS', ',', 'Spring', 'AOP', ',', 'Mule', 'ESB', ',', 'Apache', 'CXF', ',', 'Maven', ',', 'TC', 'server', ',', 'Mule', 'Server', ',', 'Jenkins', ',', 'JUnits', 'and', 'EasyMocks', ',', 'Postgres', '.', 'Syntel', 'INC', '(', 'Sep', '2008', '–', 'Nov', '2012', ')', 'PayPal', ',', 'San', 'Jose', ',', 'CA', 'Onshore', 'team', ',', 'San', 'Jose', ',', 'USA', 'Email', 'Contact', 'Management', '(', 'May', '12', '–', 'Nov', '12', ')', 'Responsibilities', ':', 'As', 'a', 'Senior', 'Software', 'Engineer/', 'Tech', 'Lead', 'my', 'responsibilities', 'in', 'this', 'project', 'are', ':', 'Coding', 'the', 'business', 'tier', 'which', 'includes', 'the', 'business', 'objecs', 'and', 'delegates', 'using', 'JDK', '1.5', ',', 'Spring', 'Core', ',', 'Spring', 'DAO', 'components', 'Environment', ':', 'Java', '5', ',', 'JSP', ',', 'Apache', 'Struts', '1.1', ',', 'Spring', 'core', ',', 'Spring', 'DAO', ',', 'Spring', 'JMS', ',', 'XML', ',', 'JMS', ',', 'DB2', ',', 'Oracle', '9i', ',', 'PL', '/', 'SQL', ',', 'DB', 'Objects', 'like', 'Stored', 'Procedures', ',', 'WebSphere', 'Application', 'Server', '6.1', ',', 'Log4J', ',', 'UNIX', 'Box', ',', 'AJAX', 'American', 'Express', ',', 'Phoenix', 'AZ', 'Offshore', 'Team', 'Chennai', ',', 'India', 'GCPSS', 'SELF', 'SERVICE', '(', 'January', '10', '–', 'April', '12', ')', 'Responsibilities', ':', 'As', 'a', 'Senior', 'Software', 'Engineer/', 'Tech', 'Lead', 'my', 'responsibilities', 'in', 'this', 'project', 'are', ':', 'Design', 'and', 'Coding', 'of', 'web', 'components', 'using', 'JSP', ',', 'Struts', '1.2', ',', 'Spring', '2.5', ',', 'JDK', '1.5', ',', 'Spring', 'Core', ',', 'Spring', 'DAO', 'Design', 'and', 'Coding', 'of', 'communication', 'module', 'with', 'MQ', 'Series', 'using', 'Spring', 'JMS', 'American', 'Express', ',', 'Phoenix', 'AZ', 'Offshore', 'Team', 'Chennai', ',', 'India', 'ONLIE', 'MERCHANT', 'SERICES', '(', 'September', '08', 'to', 'December', '09', ')', 'Responsibilities', ':', 'As', 'a', 'Senior', 'Software', 'Engineer/', 'Tech', 'Lead', 'my', 'responsibilities', 'in', 'this', 'project', 'are', ':', 'Managing', 'a', 'development', 'team', 'of', '5', 'members', 'as', 'Technical', 'Team', 'lead', 'and', 'working', 'withTechnical', 'Architect', '.', 'Oracle', 'financial', 'services', '(', 'i', 'flex', 'Ltd', ')', '(', '2006', ' ', '2007', ')', 'Mizuho', 'Corporate', 'Bank', ',', 'Japan', 'Onshore', 'Team', 'Tokyo', ',', 'Japan', 'SYNDICATED', 'LOAN', 'ACTIVITIES', '(', 'April', '08', 'to', 'July', '08', ')', 'Responsibilities', ':', 'Worked', 'as', 'Onshore', 'Coordinator', 'to', 'coordinate', 'the', 'developement', 'and', 'project', 'planning', 'activities', 'between', 'Onsite', 'team', 'in', 'Japan', '.', 'Deutsche', 'bank', ',', 'Baltimore', ',', 'MD', 'Offshore', 'Team', 'Chennai', ',', 'India', 'DBPWM', '(', 'PRIVATE', 'WEALTH', 'MANAGEMENT', ')', '(', 'July', '07', 'to', 'February', '08', ')', 'Responsibilities', ':', 'Step', 'by', 'step', 'migration', 'of', 'Weblogic', 'server', 'from', 'version', '6.1', 'to', '8.1', 'and', 'then', 'to', 'Weblogic', '9.1', 'Citigroup', 'Private', 'Bank', ',', 'Singapore', 'Offshore', 'Team', 'Chennai', ',', 'India', 'RAPIDS', '(', 'RECORD', 'MANAGEMENT', 'SYSTEM', ')', '(', 'January', '07', 'to', 'June', '07', ')', 'Responsibilities', ':', 'As', 'a', 'Senior', 'Software', 'Engineer', 'my', 'responsibilities', 'in', 'this', 'project', 'are', ':', 'Design', 'and', 'Coding', 'of', 'User', 'Interfaces', 'using', 'J2EE', ',', 'JSP', ',', 'Servlet', ',', 'Hibernate', ',', 'Documentum', 'API', '’s', ',', 'JDK', '1.3', ',', 'SQL', '/', 'PLSQLinDB2', '.', 'CSC', '(', 'Covansys', 'Pvt', 'Ltd', ')', '(', '2006', '  ', '2006', ')', 'Ohio', 'State', ',', 'Ohio', 'Offshore', 'Team', 'Chennai', ',', 'India', 'EMPLOYER', 'RESOURCE', 'INFORMATION', 'CENTER', 'Responsibilities', ':', 'As', 'a', 'Senior', 'Software', 'Engineer', 'my', 'responsibilities', 'in', 'this', 'project', 'are', ':', 'Coding', 'the', 'User', 'Interfaces', 'using', 'JSP', ',', 'JSF', ',', 'JDK', '1.4', 'APIs', 'and', 'SQL', '/', 'PLSQLinDB2', '.', 'STANDARD', 'CHARTERED', 'BANK', 'Onshore', 'Team', 'Dubai', ',', 'UAE', 'REWARDS', '(', 'August', '05', 'to', 'March', '06', ')', 'Responsibilities', ':', 'Coding', 'the', 'User', 'Interfaces', 'using', 'JSP', 'and', 'Tacheon', 'framework', ',', 'Java', 'mail', 'API', '’s', 'and', 'SQL', '/', 'PLSQL', 'in', 'Oracle', '.', 'ORCHIDSOFT', '(', 'P', ')', 'LTD', '(', '2003', '  ', '2005', ')', 'Offshore', 'Team', 'Chennai', ',', 'India', 'May', '2003', 'July', '2005', 'INSURANCEFORUSA', 'Responsibilities', ':', 'Coding', 'the', 'User', 'Interfaces', 'using', 'JSP', ',', 'Servlets', ',', 'JDK', '1.3', ',', 'SQL', '/', 'PLSQL', 'in', 'Oracle', '.']
# # In[37]:


def to_matrix_senti(padd, n):
    return [padd[i:i+n] for i in range(0, len(padd), n)]
padd_to_2d_senti = list(to_matrix_senti(end_1,31))
# print(len(padd_to_2d_senti))

# @app.route('/',methods=['POST','GET'])
# def padd():
#     print(padd_to_2d_senti)
#     return 'jpt'
# app.run()
# # In[38]:


new_matrix_senti = []
for seq in padd_to_2d_senti:
    new_seq_senti = []
    for i in range(max_len):
        try:
            new_seq_senti.append(seq[i])
        except:
            new_seq_senti.append("__PAD__")
    new_matrix_senti.append(new_seq_senti)
padd_to_2d_senti = new_matrix_senti

# # In[39]:


padd_to_2d_senti = np.array(padd_to_2d_senti)


# # In[46]:


y = model.predict([padd_to_2d_senti])

idx2tags = {0: 'I-COMPANY', 1: 'I-EMAIL', 2: 'B-LOCATION', 3: 'B-COMPANY', 4: 'B-WEBSITE', 5: 'B-COLLEGE', 6: 'B-PERSON', 7: 'B-EMAIL', 8: 'B-SKILLS', 9: 'I-COLLEGE', 10: 'I-PERSON', 11: 'B-DATE', 12: 'DOT', 13: 'I-LOCATION', 14: 'I-NUMBER', 15: 'I-UNI', 16: 'B-UNI', 17: 'B-EDUCATION', 18: 'B-NUMBER', 19: 'I-DATE', 20: 'COMMA', 21: 'O', 22: 'I-EDUCATION', 23: 'I-SKILLS'}
# # i = 8

p = np.argmax(y[:32], axis=-1)
# print(p)
flat_list_te = padd_to_2d_senti[:32]
flat_list_test = [item for sublist in flat_list_te for item in sublist]
# print(flat_list_test)
# flat_list = [item for sublist in y_te[i] for item in sublist]
flat_list_pred = [item for sublist in p for item in sublist]
# print(flat_list_pred)
flat_list_pred_value_in_words = []
for i in flat_list_pred:
    flat_list_pred_value_in_words.append(idx2tags[i])
# print(len(flat_list_pred_value_in_words))

# print(p)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
test_pred = dict(zip(flat_list_test,flat_list_pred_value_in_words))

# test_pred = [{'TEST':test,'PRED':pred} for test,pred in zip(flat_list_test,flat_list_pred_value_in_words)]
test_predicted = json.dumps(test_pred)
@app.route('/',methods=['POST','GET'])
def padd():
    return test_predicted
app.run()
# print(test_predicted)



