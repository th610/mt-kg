## Requirements
# python 3.10
# numpy
import pickle

with open('all_movies.pkl', 'rb') as fid:
    all_mg = pickle.load(fid, encoding='latin1')
    # latin1 is important here

# all_mg is a dictionary of MovieGraph objects
# indexed by imdb unique movie identifiers
num_movies = len(all_mg.keys())
print('Found {} movies with graphs'.format(num_movies))


mg = all_mg['tt0109830']  # Forrest Gump
print('Selected movie: {}'.format(mg.imdb_key))
print()
print('Cast in this movie:')
for character in mg.castlist:
    print(character['chid'], character['name'])

# mg.clip_graphs is a list of ClipGraph objects
print()
print('Selected one clip graph')
cg = mg.clip_graphs[0]
cg.pprint()


