import sys, os, json, re, random
sys.path.append('./../')

from docopt import docopt

from libs import data_manipulator

import collections
import nltk

def read_strp_lines(file):
    with open (file) as f_in:
        return [line.strip() for line in f_in]

def is_excluded(w1, w2, exclude_words):
        for excl in exclude_words:
            if w1 in excl and w2 in excl:
                return True

        return False

def uknown_words(words):
    return [w for w in words if w in vocab]

def synonym_from_two_lists(list_a, list_b):
    return [(list_a[i], list_b[i], 'entailment') for i in range(len(list_a))]

def compatible_to_first(words1, words2, exclude_words=[], symmetric=True):
    results = []
    for w1 in words1:
        for w2 in words2:
            if not is_excluded(w1, w2, exclude_words):
                results.append((w1, w2, 'entailment'))
                if symmetric:
                    results.append((w2, w1, 'entailment'))

    return results

def all_compatible(words):
    results = []
    for w1 in words:
        for w2 in words:
            if w1 != w2:
                results.append((w1, w2, 'entailment'))

    return results

def all_incompatible(words, exclude_words=[]):

    results = []
    for w1 in words:
        for w2 in words:
            if w1 != w2 and not is_excluded(w1, w2, exclude_words):
                results.append((w1, w2, 'contradiction'))

    return results

def incompatible_to_first(words1, words2, exclude_words=[], symmetric=True):
    results = []
    for w1 in words1:
        for w2 in words2:
            if not is_excluded(w1, w2, exclude_words):
                results.append((w1, w2, 'contradiction'))
                if symmetric:
                    results.append((w2, w1, 'contradiction'))

    return results

def incompatible_pairs(pairs, symmetric=True):
    results = []
    for first_words, second_words in pairs:
        for w1 in first_words.split(','):
            for w2 in second_words.split(','):
                results.append((w1, w2, 'contradiction'))
                if symmetric:
                    results.append((w2, w1, 'contradiction'))

    return results

def synonyms():
    two_way_synonyms1 = 'poisonous,relies on,hotel,detests,colossal,cabs,cab,father,sofa,near,roads,road,garbage,mom,speaking,shouts,shout,coworker,demolish,announce,dead,starts,quickly,noisy,no one,angry,angry,awful,dangerous,beautiful,fantastic,enormous,wealthy,famous,stupid,interesting,a lot of'.split(',')
    two_way_synonyms2 = 'toxic,depends on,inn,hates,enormous,taxis,taxi,dad,couch,close to,streets,street,trash,mother,talking,yells,yell,colleague,destroy,declare,lifeless,begins,rapidly,loud,nobody,furious,mad,terrible,risky,pretty,wonderful,huge,rich,well-known,dumb,fascinating,plenty of'.split(',')

    syn_group1 = 'happy,pleased,delighted,joyful,glad'.split(',')
    syn_group2 = 'sad,miserable,unhappy'.split(',')
    syn_group3 = 'excellent,outstanding,phenomenal'.split(',')
    syn_group4 = 'clever,smart,intelligent'.split(',')
    syn_group5 = 'tiny,small,little'.split(',')

    replace_only_first = 'speech,timid,assistance,allow,watchful,accident,chefs,chef,gifts,gift,speaks,speak,destroys,demolish,destroy,injured,different,begin,quickly,rapidly,silent,crazy,ancient,a different'.split(',')
    not_replace_second = 'talk,shy,aid,permit,alert,crash,cooks,cook,presents,present,talks,talk,ruins,ruin,ruin,hurt,distinct,start,fast,fast,quiet,mad,old,the same'.split(',')

    replace_any = synonym_from_two_lists(two_way_synonyms1, two_way_synonyms2)
    replace_any.extend(synonym_from_two_lists(two_way_synonyms2, two_way_synonyms1))
    replace_any.extend(all_compatible(syn_group1))
    replace_any.extend(all_compatible(syn_group2))
    replace_any.extend(all_compatible(syn_group3))
    replace_any.extend(all_compatible(syn_group4))
    replace_any.extend(all_compatible(syn_group5))

    replace_first = synonym_from_two_lists(replace_only_first, not_replace_second)
    replace_second = synonym_from_two_lists(not_replace_second, replace_only_first)

    return ('synonyms', replace_first, replace_second, replace_any)

def countries():
    countries = 'America,China,India,England,Japan,Russia,Canada,Germany,Australia,Holland,France,Israel,Spain,Brazil,Sweden,Greece,Italy,Ireland,Mexico,Switzerland,Singapore,Turkey,Ukraine,Egypt,Malaysia,Norway,Vietnam'.split(',')
    exclude_words = [set(['America', 'Canada'])]

    return ('countries', [], [], all_incompatible(countries, exclude_words=exclude_words))

def nationalities():
    exclude_words = []

    nationalities_same_singular_plural = 'Chinese,English,Japanese,Dutch,French,Spanish,Swedish,Irish,Turkish,Vietnamese'.split(',')
    nationalities_different_plural_singular = 'Russian,Canadian,German,Australian,Israeli,Brazilian,Greek,Italian,Mexican,Ukrainian,Egyptian,Norwegian,Indonesian'.split(',')
    nationalities_different_plural_plural = 'Russians,Canadians,Germans,Australians,Italians,Mexicans,Ukrainians,Norwegians,Indonesians'.split(',')
    
    replace_any = all_incompatible(nationalities_same_singular_plural)
    replace_any.extend(all_incompatible(nationalities_different_plural_plural))
    replace_any.extend(all_incompatible(nationalities_different_plural_singular))

    replace_first = incompatible_to_first(nationalities_different_plural_singular, nationalities_same_singular_plural, symmetric=False)
    replace_first.extend(incompatible_to_first(nationalities_different_plural_plural, nationalities_same_singular_plural, symmetric=False))

    replace_second = incompatible_to_first(nationalities_same_singular_plural, nationalities_different_plural_singular, symmetric=False)
    replace_second.extend(incompatible_to_first(nationalities_same_singular_plural, nationalities_different_plural_plural, symmetric=False))

    return ('nationalities', replace_first, replace_second, replace_any)

def colors():
    colors = 'red,blue,yellow,purple,green,brown,grey,black,white,turquoise,violet,beige,silver,pink'.split(',')
    exclude_pairs = [set(['turquoise', 'blue']), set(['violet', 'purple', 'pink'])]

    replace_any = all_incompatible(colors, exclude_pairs)
    replace_first = incompatible_to_first(colors, ['orange'], symmetric=False)
    replace_second = incompatible_to_first(['orange'], colors, symmetric=False)

    return ('colors', replace_first, replace_second, replace_any)

def antonyms_adj_adv():
    antonyms_replace_any = [
        ('slow','rapid,quick'),('slowly','rapidly,quickly'),('silently,quietly','loudly'),('calm','angry,furious'),
        ('pretty,beautiful','ugly'),('intelligent,clever,smart','stupid'),('small,little,tiny','big,giant,huge'),
        ('famous,well-known','unknown'),('rich','poor'),('dead','alive'),('awful,terrible','wonderful,fantastic'),
        ('ancient,antique,old-fashioned','modern'),('best','worst'),('better','worse'),('capable','incapable'),('careful','careless'),
        ('carefully','carelessly'),('expensive,costly','cheap'),('cheerful,happy,delighted,joyful,glad','sad,unhappy,sorrowful,depressed'),
        ('cloudy','sunny'),('clumsy','graceful'),('correct','wrong'),('cruel,vicious,rude','friendly'),('dangerous,unsafe','safe'),
        ('dry','wet'),('early','late'),('external','internal'),('innocent','guilty'),('healthy','unhealthy'),('vertical','horizontal'),
        ('intentionally','accidentally'),('proud','embarrassed'),('legal','illegal'),('foolish','wise'),('likely','unlikely'),('minor','major'),
        ('mature','immature'),('minimum','maximum'),('obedient','disobedient'),('permanently','temporarily'),('rude','polite'),('drunk','sober'),
        ('artificial','natural'),('pleased','displeased'),('silent','noisy,loud'),('slim,slender','fat'),('amateur','professional'),
        ('annoyed','satisfied')
    ]
    antonyms_replace_only_first = [
        ('slow,slowly','fast'),('noisy,loud','quiet'),('new','old'),('young','old'),('slim,slender','thick'),('asleep','awake'),
        ('narrow','broad'),('cold','warm,hot'),('cruel,vicious,rude','kind'),('shallow','deep'),('easy','difficult,hard'),('empty','full'),
        ('soft','hard'),('sick,ill','healthy'),('long','short'),('tight','loose'),('impatient','patient'),('temporary','permanent'),
        ('impossible','possible'),('private','public'),('fake','real'),('invisible','visible'),('complicated','simple'),('wrong','right'),
        ('a different','the same'),('hilly','flat')

    ]

    replace_any = incompatible_pairs(antonyms_replace_any)
    replace_first = incompatible_pairs(antonyms_replace_only_first, symmetric=False)
    replace_second = incompatible_pairs([(b, a) for a, b in antonyms_replace_only_first], symmetric=False)

    return ('antonyms_adj_adv', replace_first, replace_second, replace_any)

def antonyms_nn_vb():
    antonyms_replace_any = [
        ('everybody','nobody,no one'),('comfort','discomfort'),('day','night'),('daytime','nighttime'),('north','south'),
        ('injustice','justice'),('presence','absence'),('approval','disapproval'),('loves','detests,hates,dislikes'),
        ('accepts','declines'),('agree','disagree'),('approves','disapproves'),('praise','blame'),('conceal','reveal'),
        ('conceals','reveals'),('concealing','revealing'),('enters','exits'),('fail','succeed'),('failing','succeeding'),
        ('remembered','forgot'),('remembers','forgets'),('ascend','descend'),('include','exclude'),('including','excluding'),
        ('ascends','descends'),('ascending','descending'),('attacks','defends'),('attacking','defending'),('pleases','displeases'),
        ('amuses','bores'),('annoy','satisfy'),('good luck','bad luck'),('sell','buy'),('sells','buys'),('buying','selling'),
        ('compliment','insult'),('noon','morning,midnight'),('morning','midnight'),('moon','sun'),('whisper','shout,yell,scream'),
        ('whispers','shouts,yells,screams')
    ]

    antonyms_replace_only_first = [
        ('enemy','ally,friend'),('west','east'),('loss','win'),('departure','arrival'),('start','end,finish'),('accept','decline'),
        ('bless','curse'),('build','destroy'),('builds','destroys'),('cannot,can not','can'),('enter','exit'),('remember','forget'),
        ('defend','attack'),('fixes','breaks'),('bought','sold')
    ]

    replace_any = incompatible_pairs(antonyms_replace_any)
    replace_first = incompatible_pairs(antonyms_replace_only_first, symmetric=False)
    replace_second = incompatible_pairs([(b, a) for a, b in antonyms_replace_only_first], symmetric=False)

    return ('antonyms_nn_vb', replace_first, replace_second, replace_any)

def antonyms_other():
    antonyms_replace_any = [
        ('far from,far away from','close to,near'),('inside','outside'),('always','never,sometimes,often'),('never','sometimes,often'),
        ('forward','backward'),('downwards','upwards'),('before','after'),('above','below'),('behind of','in front of')
    ]

    antonyms_replace_only_first = [
        ('without','with'),('more','less')
    ]

    replace_any = incompatible_pairs(antonyms_replace_any)
    replace_first = incompatible_pairs(antonyms_replace_only_first, symmetric=False)
    replace_second = incompatible_pairs([(b, a) for a, b in antonyms_replace_only_first], symmetric=False)

    return ('antonyms_other', replace_first, replace_second, replace_any)

def numbers():
    numbers_written = 'two,three,four,five,six,seven,eight,nine,ten,eleven,twelve'.split(',')
    numbers_digits = '2,3,4,5,6,7,8,9,10,11,12'.split(',')

    exclude_pairs = [set([numbers_written[i], numbers_digits[i]]) for i in range(len(numbers_written))]

    incompatible_written_numbers = all_incompatible(numbers_written)
    incompatible_digits = all_incompatible(numbers_digits)
    incompatible_written_premise = incompatible_to_first(numbers_written, numbers_digits, exclude_pairs, False)
    incompable_digits_premise = incompatible_to_first(numbers_digits, numbers_written, exclude_pairs, False)

    synonym_numbers = synonym_from_two_lists(numbers_written, numbers_digits)
    synonym_numbers_reversed = synonym_from_two_lists(numbers_digits, numbers_written)

    replace_first = synonym_numbers + incompatible_written_premise
    replace_second = synonym_numbers_reversed + incompable_digits_premise
    replace_any = incompatible_written_numbers + incompatible_digits


    return ('numbers', replace_first, replace_second, replace_any)


def fruits():
    singular = 'watermelon,strawberry,raspberry,pineapple,pear,apple,banana,blueberry,cherry,coconut,fig,grape,lemon,mango'.split(',')
    plural = 'watermelons,strawberries,raspberries,pineapples,pears,apples,apricots,bananas,blueberries,cherries,coconuts,grapes,lemons,limes,mangos,peaches'.split(',')
    dont_replace_singular = 'lime,peach'.split(',')

    replace_any = all_incompatible(singular)
    replace_any.extend(all_incompatible(plural))

    replace_first = incompatible_to_first(singular, dont_replace_singular, symmetric=False)
    replace_second = incompatible_to_first(dont_replace_singular, singular, symmetric=False)

    replace_first.extend(compatible_to_first(singular, ['fruit'], symmetric=False))
    replace_first.extend(compatible_to_first(plural, ['fruits'], symmetric=False))

    return ('fruits', replace_first, replace_second, replace_any)

def vegetables():
    singular = 'zucchini,tomato,potato,onion,eggplant,cucumber,celery,carrot,avocado'.split(',')
    plural = 'tomatoes,potatoes,onions,eggplants,cucumbers,carrots,avocados'.split(',')
    dont_replace_singular = ['pumpkin']
    dont_replace_plural = ['pumpkins']

    replace_any = all_incompatible(singular)
    replace_any.extend(all_incompatible(plural))

    replace_first = incompatible_to_first(singular, dont_replace_singular, symmetric=False)
    replace_first.extend(incompatible_to_first(plural, dont_replace_plural, symmetric=False))

    replace_second = incompatible_to_first(dont_replace_singular, singular, symmetric=False)
    replace_second.extend(incompatible_to_first(dont_replace_plural, plural, symmetric=False))

    replace_first.extend(compatible_to_first(singular, ['vegetable'], symmetric=False))
    replace_first.extend(compatible_to_first(plural, ['vegetables'], symmetric=False))
    
    return ('vegetables', replace_first, replace_second, replace_any)

def drinks():
    alcohol = 'beer,champagne,whisky,wine,gin,vodka,tequila,cider'.split(',')
    plural = ['beers', 'wines']
    non_alcohol = 'hot chocolate,lemonade,coka-cola,coke,sprite'.split(',')
    replace_other_only = 'coffee,espresso,juice,tea'.split(',')
    exclude_words = [set(['coka-cola', 'coke', 'lemonade']), set(['lemonade', 'sprite'])]
    replace_any = all_incompatible(alcohol)
    replace_any.extend(all_incompatible(plural))
    replace_any.extend(all_incompatible(non_alcohol, exclude_words=exclude_words))

    replace_first = incompatible_to_first(non_alcohol, replace_other_only, symmetric=False)
    replace_second = incompatible_to_first(replace_other_only, non_alcohol, symmetric=False)

    return ('drinks', replace_first, replace_second, replace_any)


def fastfoods():
    singular = 'kebab,hamburger,cheeseburger,sandwich,taco,pizza'.split(',')
    plural = 'kebabs,french fries,chicken nuggets,onion rings,fish and chips,falafel,pizzas'.split(',')

    replace_any = all_incompatible(singular)
    replace_any.extend(all_incompatible(plural))

    return ('fastfood', [], [], replace_any)

def movements():
    thrird_person = 'swims to,drives to,strolls to,runs to,rides to'.split(',')
    infinitiv = 'fly to,drive to,stroll to,ride to,crawl to,run to'.split(',')

    replace_other = 'walk to, walks to'.split(',')
    replace_any = all_incompatible(thrird_person)
    replace_any.extend(all_incompatible(infinitiv))

    replace_first = incompatible_to_first(thrird_person, ['walks to'], symmetric=False)
    replace_second = incompatible_to_first(['walks to'], thrird_person, symmetric = False)

    replace_first.extend(incompatible_to_first(infinitiv, ['walk to'], symmetric=False))
    replace_second.extend(incompatible_to_first(['walk to'], infinitiv, symmetric=False))


    return ('movements', replace_first, replace_second, replace_any)

def materials():
    group1 = 'brick,cement,plastic,sand,stone,wood,titanium,bronze,copper,aluminium'.split(',')
    group1_replace_other=['glass']
    group_metals = 'titanium,bronze,copper,aluminium'.split(',')

    group2 = 'leather,nylon,wool'.split(',')
    group_2_replace_other = ['cotton']

    replace_any = all_incompatible(group1)
    replace_any.extend(all_incompatible(group2))

    replace_first = incompatible_to_first(group1, group1_replace_other, symmetric=False)
    replace_first.extend(incompatible_to_first(group2, group_2_replace_other, symmetric=False))

    replace_second = incompatible_to_first(group1_replace_other, group1, symmetric=False)
    replace_second.extend(incompatible_to_first(group_2_replace_other, group2, symmetric=False))

    replace_any.extend(compatible_to_first(group_metals, ['metal'], symmetric=False))

    return ('materials', replace_first, replace_second, replace_any)

def planets():
    planets = 'Venus,Earth,Mars,Jupiter,Saturn,Uranus,Neptune,Pluto'.split(',')
    replace_any = all_incompatible(planets)
    replace_first = incompatible_to_first(planets, ['Mercury'], symmetric=False)
    replace_second = incompatible_to_first(['Mercury'], planets, symmetric=False)

    return ('planets', replace_first, replace_second, replace_any)

def verb_at():
    base_words1 = 'sit,stand,lie'.split(',')
    base_words2 = 'sits,stands,lies'.split(',')

    with_at1 = 'sit at,stand at,walk at,run at,climb at,sleep at'.split(',')
    with_at2 = 'sits at,stands at,lies at,walks at,runs at,crawls at,climbs at,sleeps at'.split(',')
    exclude_pairs = [set(['sleeps at','lies at']), set(['sleeps at', 'sits at']), set(['sleeps at', 'stands at'])]
    exclude_pairs.extend([set(['lies at', 'crawls at']), set(['sleep at', 'sit at']), set(['sleep at', 'stand at'])])

    replace_other_only ='jump at,jumps at,'

    replace_any = all_incompatible(base_words1)
    replace_any.extend(all_incompatible(base_words2))
    replace_any.extend(all_incompatible(with_at1, exclude_pairs))
    replace_any.extend(all_incompatible(with_at2, exclude_pairs))

    replace_first = incompatible_to_first(with_at1, ['jump at'], symmetric=False)
    replace_second = incompatible_to_first(['jump at'], with_at1, symmetric=False)

    replace_first.extend(incompatible_to_first(with_at2, ['jumps at'], symmetric=False))
    replace_second.extend(incompatible_to_first(['jumps at'], with_at2, symmetric=False))

    return ('at-verbs', replace_first, replace_second, replace_any)

def rooms():
    words = 'basement,bathroom,bedroom,living room,kitchen,dining room,prison cell,cellar,garage,hallway,lounge,playroom,common room,courtroom'.split(',')
    replace_other_only = 'office,classroom,lounge'.split(',')
    general_words = 'in a building,in a room'.split(',')
    exclude_words = [set(['room', 'garage'])]

    replace_any = all_incompatible(words)
    replace_first = incompatible_to_first(words, replace_other_only, symmetric=False)
    replace_second = incompatible_to_first(replace_other_only, words, symmetric=False)
    
    
    in_a_room = ['in a '+w for w in words if w not in ['playroom', 'pantry', 'suite']]
    replace_first.extend(compatible_to_first(in_a_room, general_words, exclude_words=exclude_words, symmetric=False))

    return ('rooms', replace_first, replace_second, replace_any)


def instruments():
    singular = 'french horn,didgeridoo,tuba,xylophone,violin,trumpet,saxophone,piano,oboe,harp,accordion,banjo,cello,clarinet,flute,electric guitar,acoustic guitar,harmonica'.split(',')
    plural = 'didgeridoos,tubas,xylophones,violins,trumpets,saxophones,pianos,harmonicas,accordions,banjos,bongos,cellos'.split(',')
    #words = 'instrument,'.split(',')

    replace_any = all_incompatible(singular)
    replace_any.extend(all_incompatible(plural))

    replace_first = compatible_to_first(singular, ['instrument'], symmetric=False)
    replace_first.extend(compatible_to_first(plural, ['instruments'], symmetric=False))

    return ('instruments', replace_first, [], replace_any)


def fix():
    replace_any = incompatible_pairs([('in the morning', 'at midnight')])

    return ('fix', [], [], replace_any)

def test():
    return ('test', [('a', 'HORSE', 'contradiction'), ('NOOO WAY', 'a', 'contradiction')], [('NOOO WAY', 'the', 'contradiction'), ('omelette', 'airplane', 'contradiction')], [('horse', 'omelette', 'contradiction')])

def test_out():
    #words = 'equal,distinct,different,hurt,injure,danger,risk,facts,data,dead,lifeless,deadly,mortal,decide,determine,resolve,decision,conclusion,declare,announce,decrease,reduce,happyness,joy,gladness,demolish,destroy,denial,refusal,deny,refuse,denies,refuses,destination,goal,destiny,fate,colleague,coworker,small,tiiny,shout,yell,shouts,yells,speaks,talks,speaking,talking,clever,smart,present,gift,mother,mom,bunny,rabbit,garbage,trash,shuts,closes,shop,store,sees,looks,see,look,alike,same,chef,cook,crash,accident,raise,lift,stone,rock,stones,rocks,street,road,street,roads,near,close to,couch,sofa,father,dad,tired,sleepy,taxi,cab'.split(',')
    

    #words = 'avocado,avocados,carrot,carrots,celery,celeries,chick peas,cucumber,cucumbers,eggplant,eggplants,onion,onions,pumpkin,pumpkins,paotato,potatoes,tomato,tomatoes,vegetable,vegetables,zucchini,zucchinis'.split(',')
    #words = 'accordion,accordions,banjo,banjos,bongos,cello,cellos,clarinet,drum,flute,guitar,guitars,harmonica,harmonicas,harp,harps,kettledrum,kettledrums,oboe,piano,pianos,saxophone,saxophones,trumpet,trumpets,violin,violins,xylophone,xylophones,tuba,tubas,didgeridoo,didgeridoos,acoustic guitar,french horn'.split(',')
    #datahandler = data_manipulator.DataManipulator().load()
    #datahandler.print_sents(words, 30)

    name, repl1, repl2, repl_a = instruments()
    print('repl first')
    for p, h, lbl in repl1:
        print(p, '--', h, '--', lbl)
    print()
    print('repl second')
    for p, h, lbl in repl2:
        print(p, '--', h, '--', lbl)
    print()
    print('repl any')
    for p, h, lbl in repl_a:
        print(p, '--', h, '--', lbl)
    print()

def _parse_all_summary(in_path):
    with open(in_path) as f_in:
        lines = [line.strip() for line in f_in.readlines()]
        all_pairs = [json.loads(line) for line in lines]

    parsed = [(
        pair['word_p'], pair['word_h'], pair['amount'], pair['assumed_label'],
        pair['rel_path'], pair['sents_with_word_p'], pair['sents_with_word_h'],
        pair['real_sample_count'], pair['generate_replace']
    ) for pair in all_pairs]

    return parsed


def _parse_word_pair(in_path):
    '''
    Parse the content of a wordpair.jsonl file
    '''
    with open(in_path) as f_in:
        all_samples = [json.loads(line.strip()) for line in f_in.readlines()]

    sentences = [(sample['sentence1'], sample['sentence2']) for sample in all_samples]
    replaced = [int(sample['generation_replaced']) for sample in all_samples]

    return (sentences, replaced)


def _parse_group_summary(in_path, raw=False):
    '''
    Parse the content of the <*.sjson> file.
    '''
    with open(in_path) as f_in:
        lines = [line.strip() for line in f_in.readlines()]
        all_pairs = [json.loads(line) for line in lines]

    parsed = [(
        pair['word_p'], pair['word_h'], pair['amount'], pair['assumed_label'],
        pair['rel_path'], pair['sents_with_word_p'], pair['sents_with_word_h'],
        pair['real_sample_count'] 
    ) for pair in all_pairs]

    if raw:
        return (parsed, lines)
    else:
        return parsed

def clean_group(category_dir, name, summary):

    def include_both(w1, w2, remove_set):
        return w1 in remove_set and w2 in remove_set

    remove_files = []
    keep_lines = []
    parsed, lines = _parse_group_summary(summary, raw = True)
    for i, (w1, w2, amount, lbl, rel_path, any1, any2, any3) in enumerate(parsed):
        if name == 'nationalities':
            remove_set1 = set('Australian,Canadian,English,Irish'.split(','))
            remove_set2 = set('Spanish,Argentinian,Mexican'.split(','))
            if include_both(w1, w2, remove_set1) or include_both(w1, w2, remove_set2):
                remove_files.append(rel_path)
            else:
                keep_lines.append(i)
        elif name == 'fruits':
            if (w1 == 'coconut' and w2 == 'fruit') or (w1 == 'coconuts' and w2 == 'fruits'):
                remove_files.append(rel_path)
            else:
                keep_lines.append(i)
        elif name == 'vegetables':
            invalid_p1 = set('avocado,pumpkin,tomato'.split(','))
            invalid_h1 = set(['vegetable'])
            invalid_p2 = set('avocados,pumkins,tomatoes'.split(','))
            invalid_h2 = set(['vegetables'])

            if w1 in invalid_p1 and w2 in invalid_h1:
                remove_files.append(rel_path)
            elif w1 in invalid_p2 and w2 in invalid_h2:
                remove_files.append(rel_path)
            else:
                keep_lines.append(i)

        elif name == 'fastfood':
            remove_set1 = set('cheeseburger,hamburger,sandwich'.split(','))
            remove_set2 = set('french fries,fish and chips'.split(','))
            if include_both(w1, w2, remove_set1) or include_both(w1, w2, remove_set2):
                remove_files.append(rel_path)
            else:
                keep_lines.append(i)

        elif name == 'movements':
            remove_set = set('stroll to,strolls to,walk to,walks to'.split(','))
            if include_both(w1, w2, remove_set):
                remove_files.append(rel_path)
            else:
                keep_lines.append(i)

        elif name == 'materials':
            remove_set1 = set('brick,stone'.split(','))
            remove_set2 = set('cement,sand'.split(','))
            if include_both(w1, w2, remove_set1) or include_both(w1, w2, remove_set2):
                remove_files.append(rel_path)
            else:
                keep_lines.append(i)

        elif name == 'at-verbs':
            remove_set = set('climb at,stand at,climbs at,stands at'.split(','))
            if include_both(w1, w2, remove_set):
                remove_files.append(rel_path)
            else:
                keep_lines.append(i)

        elif name == 'rooms':
            remove_set1 = set('basement,cellar'.split(','))
            keep_set1 = set('in a house,in a building'.split(','))

            remove_set2 = set('classroom,common room'.split(','))
            remove_set3 = set('common room,living room,playroom'.split(','))
            remove_set4 = set('garage,playroom'.split(','))
            remove_set5 = set('living room,playroom,dining room,kitchen'.split(','))
            remove_set6 = set('bathroom,bedroom,living room'.split(','))
            keep_set2 = set('classroom,courtroom,prison cell,garage'.split(','))
            keep_set3 = set('classroom,prison cell,garage'.split(','))
            
            if (w1 in remove_set1 and w2 not in keep_set1) or (w1 == 'lounge' and w2 in keep_set1):
                remove_files.append(rel_path)
            elif w1 in remove_set1 or w2 in remove_set1:
                remove_files.append(rel_path)
            elif include_both(w1, w2, remove_set2) or include_both(w1, w2, remove_set3) or include_both(w1,w2,remove_set4):
                remove_files.append(rel_path)
            elif (w1 == 'common room' and w2 in remove_set5) or (w2 == 'common room' and w1 in remove_set5):
                remove_files.append(rel_path)
            elif (w1 == 'prison cell' and w2 in remove_set6) or (w2 == 'prison cell' and w1 in remove_set6):
                remove_files.append(rel_path)
            elif (w1 == 'lounge' and w2 not in keep_set2) or (w2 == 'lounge' and w1 not in keep_set2):
                remove_files.append(rel_path)
            elif (w1 == 'office' and w2 not in keep_set3) or (w2 == 'office' and w1 not in keep_set3):
                remove_files.append(rel_path)
            else:
                keep_lines.append(i)

        elif name == 'colors':
            remove_set1 = set('beige,brown'.split(','))
            remove_set2 = set('grey,black'.split(','))
            remove_set3 = set('grey,white'.split(','))
            if include_both(w1, w2, remove_set1) or include_both(w1,w2,remove_set2) or include_both(w1,w2,remove_set3):
                remove_files.append(rel_path)
            else:
                keep_lines.append(i)

        else:
            keep_lines.append(i)

    print
    lines = list(set([lines[i] for i in keep_lines]))
    
    # update
    with open(summary, 'w') as f_out:
        for line in lines:
            f_out.write(line + '\n')

    return [os.path.join(category_dir, file) for file in remove_files]


def clean_group_words(directory, name, summary):

    def remove_sentences_containing(file_path, words):
        regexps = [(re.compile('\\b' + w + '\\b')) for w in words]
        keep = []
        with open(file_path) as f_in:
            for line in f_in:
                parsed = json.loads(line.strip())
                found = False
                for regexp in regexps:
                    if regexp.search(parsed['sentence1']) or regexp.search(parsed['sentence2']):
                        print('remove')
                        print('[p]', parsed['sentence1'])
                        print('[h]', parsed['sentence2'] + '\n')
                        found = True
                        break

                if not found:
                    keep.append(line)

        with open(file_path, 'w') as f_out:
            for line in keep:
                f_out.write(line)


    parsed = _parse_group_summary(summary)
    for w1, w2, amount, lbl, rel_path, any1, any2, any3 in parsed:
        if name == 'countries':
            file_path = os.path.join(directory, rel_path)
            if w1 == 'France' or w2 == 'France':
                remove_sentences_containing(file_path, ['tour de France', 'Tour de France', 'Tour De France'])

def print_bigram_fails(dataset_name, out_name, t=0):

    exclude_bigrams = set(['.',',', '(', ')', ':'])

    bigram_counts = dict()
    count = 0
    with open(os.path.realpath('../../../data/bigrams/bigram_EN_snli.dat')) as f_in:
        for line in f_in:
            splitted = line.split()
            if splitted[1] not in bigram_counts:
                bigram_counts[splitted[1]] = collections.defaultdict(int)
            bigram_counts[splitted[1]][splitted[2]] = int(splitted[0])

    dataset_dir = os.path.dirname(dataset_name)
    with open(dataset_name) as f_in:
        lines = [line.strip().split(' ') for line in f_in.readlines()]

    categories = [(line[0], line[3]) for line in lines]
    for name, path in categories:
        category_dir = os.path.join(dataset_dir, name)

        parsed = _parse_group_summary(os.path.join(category_dir, 'SUMMARY.sjson'))
        for w1, w2, amount, lbl, rel_path, any1, any2, any3 in parsed:
            print('## Check:', name, '>>', rel_path)
            count_total = 0
            keep_samples = []
            not_keep_samples = []

            skip = False
            #if len(w1.split(' ')) > 1 or len(w2.split(' ')) > 1:
            #    print('# Skip:', w1 , ' -- ', w2)
            #    skip = True


            with open(os.path.join(category_dir, rel_path)) as f_in:
                if skip:
                    keep_samples = f_in.readlines()
                else:
                    for line in f_in:
                        count_total += 1
                        sample = json.loads(line.strip())
                        if sample['generation_replaced'] == '1':
                            sent = sample['sentence2']
                            replaced_word = w2
                        else:
                            sent = sample['sentence1']
                            replaced_word = w1

                        tokenized = nltk.word_tokenize(sent)
                        index = -1
                        index_first = -1
                        index_last = -1
                        multi_word1 = None
                        multi_word2 = None
                        try:
                            splitted_replaced_word = replaced_word.split(' ')
                            if len(splitted_replaced_word) > 1:
                                multi_word1 = splitted_replaced_word[0]
                                multi_word2 = splitted_replaced_word[-1]
                                index_first = tokenized.index(splitted_replaced_word[0])
                                index_last = tokenized.index(splitted_replaced_word[-1])
                            else:
                                index = tokenized.index(replaced_word)
                        except Exception:
                            #print('NOT FOUND',replaced_word,  tokenized)
                            index = -1
                            index_first = -1
                            index_last = -1

                        bigrams = []
                        if index > 0:
                            bigrams.append((tokenized[index - 1], replaced_word))
                        if index < len(tokenized) - 1 and index != -1:
                            bigrams.append((replaced_word, tokenized[index + 1]))

                        if index_first > 0:
                            bigrams.append((tokenized[index_first - 1], multi_word1))
                        if index_last < len(tokenized) - 1 and index_last != -1:
                            bigrams.append((multi_word2, tokenized[index_last + 1]))

                        keep = True
                        for bigram in bigrams:
                            b1 = bigram[0].lower()
                            b2 = bigram[1].lower()

                            if b1 not in exclude_bigrams and b2 not in exclude_bigrams:
                                counts = 0
                                if b1 in bigram_counts:
                                    counts = bigram_counts[b1][b2]
                                else: 
                                    counts = 0

                                if counts <= t:
                                    keep = False
                                    break

                        if keep:
                            keep_samples.append(line)
                        else:
                            not_keep_samples.append(line)


            # write out valid samples
            print(name, '>', rel_path, 'total:', count_total, 'keep:', len(keep_samples))
            current_dir = os.path.join(out_name, name)
            if not os.path.exists(current_dir):
                os.makedirs(current_dir)

            if len(keep_samples) > 0:
                with open(os.path.join(current_dir, rel_path), 'w') as f_out:
                    for line in keep_samples:
                        f_out.write(line)

            if len(not_keep_samples) > 0:
                with open(os.path.join(current_dir, 'removed-' + rel_path), 'w') as f_out:
                    for line in not_keep_samples:
                        f_out.write(line)

def clean_words(dataset_name):
    dataset_dir = os.path.dirname(dataset_name)
    with open(dataset_name) as f_in:
        lines = [line.strip().split(' ') for line in f_in.readlines()]

    categories = [(line[0], line[3]) for line in lines]
    for name, path in categories:
        category_dir = os.path.join(dataset_dir, name)
        clean_group_words(category_dir, name, os.path.join(category_dir, path), t=10)


def clean(dataset_name):
    dataset_dir = os.path.dirname(dataset_name)
    with open(dataset_name) as f_in:
        lines = [line.strip().split(' ') for line in f_in.readlines()]

    categories = [(line[0], line[3]) for line in lines]
    for name, path in categories:
        category_dir = os.path.join(dataset_dir, name)
        remove_files = clean_group(category_dir, name, os.path.join(category_dir, path))
        
        for file in remove_files:
            try:
                os.remove(file)
            except FileNotFoundError:
                print('not found:', file)

def clean_filtered(dataset_name):
    dataset_dir = os.path.dirname(dataset_name)
    with open(dataset_name) as f_in:
        lines = [line.strip().split(' ') for line in f_in.readlines()]

    categories = [(line[0], line[3]) for line in lines]
    for name, path in categories:
        category_dir = os.path.join(dataset_dir, name)
        parsed = _parse_all_summary(os.path.join(category_dir, 'SUMMARY.sjson'))
        
        write_out = []
        for w1, w2, amount, lbl, rel_path, swp, swh, real_samples, generation in parsed:
            if os.path.exists(os.path.join(category_dir, rel_path)):
                with open(os.path.join(category_dir, rel_path)) as f_in:
                    count = len(f_in.readlines())
                if count > 0:
                    write_out.append({
                        'word_p': w1,
                        'word_h': w2,
                        'amount': count,
                        'assumed_label': lbl,
                        'rel_path': rel_path,
                        'sents_with_word_p': swp,
                        'sents_with_word_h': swh,
                        'real_sample_count': real_samples,
                        'generate_replace': generation
                    })  

        # overwrite
        with open(os.path.join(category_dir, 'SUMMARY.sjson'), 'w') as f_out:
            for line in write_out:
                f_out.write(json.dumps(line) + '\n')



def sort_data(dataset_name, out_path):
    dataset_dir = os.path.dirname(dataset_name)
    with open(dataset_name) as f_in:
        lines = [line.strip().split(' ') for line in f_in.readlines()]

    categories = [(line[0], line[3]) for line in lines]

    premise_dict = collections.defaultdict(lambda:  [])

    for name, path in categories:
        category_dir = os.path.join(dataset_dir, name)
        parsed = _parse_all_summary(os.path.join(category_dir, 'SUMMARY.sjson'))
        
        print('# CATEGORY:', name)

        for w1, w2, amount, lbl, rel_path, swp, swh, real_samples, generation in parsed:
            wp_path = os.path.join(category_dir, rel_path)
            sentences, replaced_sent_idx = _parse_word_pair(wp_path)
            print('## sort in:', w1, '--', w2)
            for i in range(len(sentences)):
                premise, hypothesis = sentences[i]
                replaced = replaced_sent_idx[i]

                # only check for replaced premises
                if replaced == 1:
                    premise_dict[premise].append((hypothesis, name, w1, w2, lbl))

    # filter out duplicates
    print('Filter out duplicates ...')
    count = 0
    for premise, all_hyps in premise_dict.items():
        contains = set()
        keep = []
        for params in all_hyps:
            if params[0] not in contains:
                contains.add(params[0])
                keep.append(params)
            else:
                count += 1

        premise_dict[premise] = keep
    print('Filtered out:', count)

    # give summary
    count_dict = collections.defaultdict(int)
    for premise, all_hyps in premise_dict.items():
        count_dict[str(len(all_hyps))] += 1

    sorted_summary = sorted([(size, amount) for size, amount in count_dict.items()], key=lambda x: -int(x[0]))
    for size, amount in sorted_summary:
        print('premise containing', size, 'hypothesis:', amount)

    # write out

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    contents = []
    total = len(premise_dict)
    for i, (premise, all_hyps) in enumerate(premise_dict.items()):
        filename = str(i) + '.jsonl'
        file_path = os.path.join(out_path, filename)

        lines = [
            json.dumps({
                'sentence1': premise,
                'sentence2': hyp,
                'gold_label': label,
                'replaced1': w_premise,
                'replaced2': w_hypothesis,
                'category': group_name 
            }) + '\n'
            for hyp, group_name, w_premise, w_hypothesis, label in all_hyps
        ]

        contents.append((filename, [(group, w_premise, w_hypothesis) for any1, group, w_premise, w_hypothesis, any2 in all_hyps]))
        print('Write out:', i, '/', total)
        with open(file_path, 'w') as f_out:
            for line in lines:
                f_out.write(line)

    content_path = os.path.join(out_path, 'CONTENTS.jsonl')
    with open(content_path, 'w') as f_out:
        for filename, file_contents in contents:
            f_out.write(json.dumps({
                'filename': filename,
                'contents': [{
                    'group': group,
                    'w1': w_premise,
                    'w2': w_hypothesis
                } for group, w_premise, w_hypothesis in file_contents]    
            }) + '\n')




def summary_sorted(sorted_name):
    with open(sorted_name) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]

    # filter anything less than 5
    parsed = [item for item in parsed if len(item['contents']) >= 5]
    result_dict = collections.defaultdict(lambda: collections.defaultdict(int))

    for item in parsed:
        contents = item['contents']
        groups = collections.Counter([content_item['group'] for content_item in contents])
        for key, amount in groups.most_common():
            result_dict[key][amount] += 1

    # print out
    for group in result_dict:
        for counts in result_dict[group]:
            print(group, 'having:', counts, '>', result_dict[group][counts])





            

def summary(dataset_name):
    dataset_dir = os.path.dirname(dataset_name)
    with open(dataset_name) as f_in:
        lines = [line.strip().split(' ') for line in f_in.readlines()]

    categories = [(line[0], line[3]) for line in lines]
    for name, path in categories:
        category_dir = os.path.join(dataset_dir, name)

        category_count_samples = 0
        category_count_pairs = 0
        parsed = _parse_all_summary(os.path.join(category_dir, 'SUMMARY.sjson'))
        print('##', name)
        for w1, w2, amount, lbl, rel_path, swp, swh, real_samples, generation in parsed:
            category_count_pairs += 1
            category_count_samples += amount

            print(w1, '--', w2, '--', lbl, '>', amount)

        print('#', name, 'total pairs:', category_count_pairs, ', total samples:', category_count_samples)


def grep_dataset(sorted_name, out_name):
    random.seed(9)
    MIN_HYP_AMOUNT = 5
    stop_amount = 10000 // 18


    def filter_below(filter_data, min_amount):
        return [(file, contents) for file, contents in filter_data if len(contents) >= min_amount]

    def get_list_for(group, data):

        # function helpers
        def count_cat(contents, group):
            size = len([cat for cat, any1, any2 in contents if cat == group])
            if size > MIN_HYP_AMOUNT:
                size = MIN_HYP_AMOUNT
            return size

        # function code
        counted_data = [(i, file, contents, count_cat(contents, group))for i, (file, contents) in enumerate(data)]
        relevant_data = [(i, file, contents, count) for i, file, contents, count in counted_data if count > 0]
        
        # refilter
        relevant_data = [(i, file, contents, count) for i, file, contents, count in relevant_data if len(contents) >= MIN_HYP_AMOUNT]

        if len(relevant_data) == 0:
            return (None, [])

        max_count = max([count for any1, any2, any3, count in relevant_data])

        # first consider max count the most
        return (max_count, [(i, file, contents, count) for i, file, contents, count in relevant_data if count == max_count])
        

    

    priority1 = ['antonyms_nn_vb', 'antonyms_other', 'movements', 'fastfood']
    priority2 = ['synonyms', 'planets', 'antonyms_adj_adv', 'vegetables', 'at-verbs', 'drinks']
    priority3 = ['fruits', 'rooms', 'materials','instruments', 'nationalities', 'countries', 'numbers', 'colors']
    w_counter = collections.Counter()
    grp_counter = collections.Counter()
    used_files = collections.Counter()

    grp_penalize_factor = dict()
    for g in priority1:
        grp_penalize_factor[g] = 1
    for g in priority2:
        grp_penalize_factor[g] = 500
    for g in priority3:
        grp_penalize_factor[g] = 100000

    with open(sorted_name) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]

    print('# sorted samples loaded:', len(parsed))
    data = [(item['filename'], [(content_item['group'], content_item['w1'], content_item['w2']) for content_item in item['contents']]) for item in parsed]
    data = filter_below(data, MIN_HYP_AMOUNT)
    print('keep:', len(data))

    # go through all priorities in order
    final_dataset = []
    used_groups = []
    for name, priority in [('1', priority1), ('2', priority2), ('3', priority3)]:
        print('# priority', name)
        for current_group in priority:
            # for each, until reached stop amount (or all used)
            group_end = False
            used_groups.append(current_group)

            print('>>', current_group)

            while not group_end:
                # get top samples for group
                max_count, groupdata = get_list_for(current_group, data)
                #print('sublist:', len(groupdata), 'different premise, having', max_count, 'beautiful samples for the group.')
                if grp_counter[current_group] >= stop_amount:
                    group_end = True
                    break
                elif len(groupdata) == 0:
                    group_end = True
                    print('# end due to lack of samples.')
                    break

                sample_from = []

                # penalize already occuring sentences
                groupdata = [(i, file, contents, count, used_files[file]) for i, file, contents, count in groupdata]
                #print('my nice group data', groupdata)
                least_used_sents = min([d[-1] for d in groupdata])
                #print('my least used sent', least_used_sents)
                groupdata = [d for d in groupdata if d[-1] == least_used_sents]

                #print('Only use those nice setnences', groupdata)

                # penalize already occuring words, full categories
                for (idx, file, contents, count, any1) in groupdata:
                    # get all for group
                    all_samples = [(group, w1, w2, w_counter[w1] + w_counter[w2]) for group, w1, w2 in contents]
                    
                    group_samples = [d for d in all_samples if d[0] == current_group]

                    group_keep_samples = []
                    least_word_penalty = -1
                    while len(group_keep_samples) < count:
                        remaining_items = [d[-1] for d in group_samples if d[-1] > least_word_penalty]
                        if len(remaining_items) > 0:
                            least_word_penalty = min(remaining_items)
                            group_keep_samples.extend([d for d in group_samples if d[-1] == least_word_penalty]) 
                        else:
                            break

                    if len(group_keep_samples) == 0:
                        print(idx, file, contents, count)
                        print('NOT GOOD')
                        1/0

                    diff = MIN_HYP_AMOUNT - count
                    add_sample = None
                    if diff > 0:
                        # find other categories to fill
                        other_samples = [(group, w1, w2, w_penalty, grp_counter[group] * grp_penalize_factor[group]) for group, w1, w2,w_penalty in all_samples if group != current_group]
                        
                        least_group_penalty = -1
                        other_keep_samples_1 = []
                        while len(other_keep_samples_1) < diff:
                            least_group_penalty = min([d[-1] for d in other_samples if d[-1] > least_group_penalty])
                            other_keep_samples_1.extend([d for d in other_samples if d[-1] == least_group_penalty])
                        
                        other_keep_samples_2 = []
                        least_word_penalty = -1
                        while len(other_keep_samples_2) < diff:
                            least_word_penalty = min([d[-2] for d in other_samples if d[-2] > least_word_penalty])
                            other_keep_samples_2.extend([d for d in other_samples if d != least_word_penalty])

                        # add random sample
                        # all from group (since less than needed)
                        pick_from_group = [(group, w1, w2) for group, w1, w2, _ in group_keep_samples]
                        pick_from_other = random.sample([(group, w1, w2) for group, w1, w2, any1,any2 in other_keep_samples_2], diff)
                        add_sample = (idx, file, pick_from_group + pick_from_other)
                        #print('need to add more', diff)
                    else:

                        # just use the samples for the current group
                        pick = random.sample([(group, w1, w2) for group, w1, w2, _ in group_keep_samples], MIN_HYP_AMOUNT)
                        #print('have engough')
                        add_sample = (idx, file, pick)

                    #print('added', add_sample)
                    sample_from.append(add_sample)


                # sample from them
                final_pick = random.choice(sample_from)
                final_dataset.append(final_pick)
                # update counts
                idx, file, sents = final_pick
                for group, w1, w2 in sents:
                    w_counter[w1] += 1
                    w_counter[w2] += 1
                    grp_counter[group] += 1
                used_files[file] += 1

                # update data and repeat
                data_filename, data_contents = data[idx]
                if data_filename != file:
                    print('should not happen')
                    1/0
                # remove used samples
                new_contents = []
                for dgroup, dw1, dw2 in data_contents:
                    add = True
                    for pgroup, pw1, pw2 in sents:
                        if dgroup == pgroup and dw1 == pw1 and dw2 == pw2:
                            add = False
                            break

                    if add:
                        new_contents.append((dgroup, dw1, dw2))
                data[idx] = (data_filename, new_contents)

                
            # group done4
            print('Finished with', current_group, grp_counter[current_group])
            total_samples = sum(grp_counter.values())
            total_finished = sum([grp_counter[g] for g in used_groups])
            print('currently having:', total_samples,'/ 10000', 'samples')
            missing = 10000 - total_finished

            missing_group_amount = 18 - len(used_groups)
            if missing_group_amount > 0:
                stop_amount = missing / missing_group_amount
            print(grp_counter.most_common())
            # greedily eat also other categories IF that includes many samples for that category (add to the other category some)
            

    with open(out_name, 'w') as f_out:
        print('Write to:', out_name)
        print('Done. Have: 5 x', len(final_dataset), 'samples.')
        for idx, file, contents in final_dataset:
            f_out.write(json.dumps({
                'filename': file,
                'contents': [{
                    'group': group,
                    'w1': w1,
                    'w2': w2
                } for group, w1, w2 in contents]
            }) + '\n')

    

    # go through priority 2
    # ...

    # go through remeining
    # ...

def finalize_dataset(dataset, out_path):
    with open(dataset) as f_in:
        lines = [line for line in f_in.readlines()]

    parsed = [json.loads(line.strip()) for line in lines]

    sample_dict = collections.defaultdict(lambda: [])

    for i, p in enumerate(parsed):
        sample_dict[p['sentence1']].append((i, p))

    count = 0
    with open(out_path, 'w') as f_out:
        for key in sample_dict:
            current_set = sample_dict[key]
            
            only_material_sand = True
            for i, p in current_set:
                if p['replaced1'] != 'sand':
                    only_material_sand = False
                    break
            if not only_material_sand:
                for i, p in current_set:
                    f_out.write(lines[i])
            else:
                count += 5

    print('Done. removed:', count)
    


def sample_dataset(dataset_path):
    with open(dataset_path) as f_in:
        parsed = [json.loads(line.strip()) for line in f_in.readlines()]

    sample_dict = collections.defaultdict(lambda: [])
    for p in parsed:
        sample_dict[p['category']].append((p['sentence1'], p['sentence2'], p['gold_label'], p['replaced1'], p['replaced2']))

    AMOUNT = 20

    for key in sample_dict:
        print('#', key)
        choice = random.sample(sample_dict[key], AMOUNT)
        for prem, hyp, lbl, w1, w2 in choice:
            print(w1, '--', w2, '--', lbl)
            print('[p]', prem)
            print('[h]', hyp)
            print()

    print('# SUMMARY')
    total= 0
    summary = sorted([(key, len(sample_dict[key])) for key in sample_dict], key=lambda x: -x[-1])
    for key, amount in summary:
        total += amount
        print(key, '-->', amount, 'samples')
    print('total:', total)
def main():
    args = docopt("""Create a new dataset based on the given type.

    Usage:
        dataset_creator.py create <out_name>
        dataset_creator.py test 
        dataset_creator.py show -a <amount> (-w <words>)...
        dataset_creator.py clean_simple <dataset_name>
        dataset_creator.py clean <dataset_name>
        dataset_creator.py bigrams <dataset_name> <out_name>
        dataset_creator.py clean_filtered <dataset_name>
        dataset_creator.py summary <dataset_name>
        dataset_creator.py datasort <dataset_name> <out_name>
        dataset_creator.py summary_sorted <sorted_name>
        dataset_creator.py grep_dataset <sorted_name> <out_name>
        dataset_creator.py finalize_dataset <dataset_name> <out_path>
        dataset_creator.py sample <datset_path>
    """)


    if args['test']:
        test_out()
    elif args['grep_dataset']:
        grep_dataset(args['<sorted_name>'], args['<out_name>'])
    elif args['sample']:
        sample_dataset(args['<datset_path>'])
    elif args['finalize_dataset']:
        finalize_dataset(args['<dataset_name>'], args['<out_path>'])
    elif args['datasort']:
        sort_data(args['<dataset_name>'], args['<out_name>'])
    elif args['summary_sorted']:
        summary_sorted(args['<sorted_name>'])
    elif args['summary']:
        summary(args['<dataset_name>'])
    elif args['clean_filtered']:
        clean_filtered(args['<dataset_name>'])
    elif args['show']:
        max_amount = int(args['<amount>'])
        words = args['<words>']
        datahandler = data_manipulator.DataManipulator().load()
        datahandler.print_sents(words, max_amount)
    elif args['clean_simple']:
        clean(args['<dataset_name>'])
    elif args['clean']:
        clean_words(args['<dataset_name>'])
    elif args['bigrams']:
        print_bigram_fails(args['<dataset_name>'], args['<out_name>'], t=10)
    else:
        out_name = args['<out_name>']
        all_fn = [
            #countries,
            #nationalities,
            #colors,
            #numbers,
            #antonyms_adj_adv,
            #antonyms_nn_vb,
            #antonyms_other,
            #synonyms,
            #fruits,
            #vegetables,
            #drinks,
            #fastfoods,
            #movements,
            #materials,
            #planets,
            #verb_at,
            #rooms,
            #instruments
            #test
            fix
        ]

        datahandler = data_manipulator.DataManipulator().load()

        groups = []
        for fn in all_fn:
            name, replace_w1_only, replace_w2_only, replace_any = fn()

            generated_sample_holder = datahandler.generate_simply(replace_w1_only, replace='w1')
            generated_sample_holder.merge(datahandler.generate_simply(replace_w2_only, replace='w2'))
            generated_sample_holder.merge(datahandler.generate_simply(replace_any, replace='any'))
            
            count_wordpairs, count_samples = generated_sample_holder.get_counts()
            groups.append((name, count_wordpairs, count_samples))
            directory = os.path.join(out_name, name)
            generated_sample_holder.write_summary(directory)
            generated_sample_holder.write_dataset(directory)

        with open(os.path.join(out_name, 'data.txt'), 'w') as f_out:
            for name, wordpair_count, sample_count in groups:
                f_out.write(' '.join([name, str(wordpair_count), str(sample_count)]) + '\n')

if __name__ == '__main__':
    main()
        

