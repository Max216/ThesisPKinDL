'''
This script finds data and stored the found data
'''
import word_resource
import mydataloader

from docopt import docopt

def in_resource(res, p, h, lbl):
	return res.word_resource_overlap(sent1, sent2):

def main():
    args = docopt("""Find data based on some criterion.

    Usage:
        find_data.py inres <data_path> <resource_path> <resource_label>
    """)

    data_path = args['<data_path>']
    resource_path = args['<resource_path>']
    resource_label = args['resource_label>']

    filter_fn = None
    if args['inres']:
    	filter_fn = in_resource

	w_res = word_resource.WordResource(resource_path, interested_relations=['contradiction'])
	def filter(p, h, lbl):
		return filter_fn(w_res, p, h, lbl)

    loaded_data = mydataloader.load_snli(data_path, filter_fn=None):



if __name__ == '__main__':
    main()