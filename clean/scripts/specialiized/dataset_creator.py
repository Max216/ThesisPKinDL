import sys, os, json
sys.path.append('./../')

from docopt import docopt

from libs import data_manipulator

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
        if name == 'countries':
            pass
        elif name == 'nationalities':
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





def clean(dataset_name):
    dataset_dir = os.path.dirname(dataset_name)
    with open(dataset_name) as f_in:
        lines = [line.strip().split(' ') for line in f_in.readlines()]

    categories = [(line[0], line[3]) for line in lines]
    for name, path in categories:
        category_dir = os.path.join(dataset_dir, name)
        remove_files = clean_group(category_dir, name, os.path.join(category_dir, path))
        
        for file in remove_files:
            os.remove(file)

def main():
    args = docopt("""Create a new dataset based on the given type.

    Usage:
        dataset_creator.py create <out_name>
        dataset_creator.py test 
        dataset_creator.py show -a <amount> (-w <words>)...
        dataset_creator.py clean_simple <dataset_name>
    """)


    if args['test']:
        test_out()
    elif args['show']:
        max_amount = int(args['<amount>'])
        words = args['<words>']
        datahandler = data_manipulator.DataManipulator().load()
        datahandler.print_sents(words, max_amount)
    elif args['clean_simple']:
        clean(args['<dataset_name>'])
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
            synonyms,
            fruits,
            vegetables,
            drinks,
            fastfoods,
            movements,
            materials,
            planets,
            verb_at,
            rooms,
            instruments
            #test
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
        

