import sys, os
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

def tools():
    kitchen = 'knife,fork,spoon' + 'knives,forks,spoons'
    cutting = 'knife,scissor' + 'knives,scissors'

def fruits():
    words = 'apple,apples,apricots,banana,bananas,blueberry,blueberries,berry,berries,fruits,cherry,cherries,coconut,coconuts,fig,grape,grapes,lemon,lemons,limes,mango,mangos,peaches,pear,pears,pineapple,pineapples,raspberry,raspberries,strawberry,strawberries,watermelon,watermelons'.split(',')
    replace_other_than = 'fruit,lime,peach'

def vegetables():
    words = 'avocado,avocados,carrot,carrots,celery,cucumber,cucumbers,eggplant,eggplants,onion,onions,potato,potatoes,tomato,tomatoes,vegetable,vegetables,zucchini'.split(',')
    replace_other='pumpkin,pumpkins'

def drinks():
    words = 'beer,beers,champagne,whisky,wine,wines,gin,vodka,tequila,cider'
    replace_this_only = 'alcohol'
    nonalcohol = 'hot chocolate,lemonade,soft drink,soft drinks,coka-cola,coke,sprite'.split(',')
    only_replacE_other = 'coffee,espresso,juice,tea'
def test():
    return ('test', [('a', 'HORSE', 'contradiction'), ('NOOO WAY', 'a', 'contradiction')], [('NOOO WAY', 'the', 'contradiction'), ('omelette', 'airplane', 'contradiction')], [('horse', 'omelette', 'contradiction')])

def test_out():
    #words = 'equal,distinct,different,hurt,injure,danger,risk,facts,data,dead,lifeless,deadly,mortal,decide,determine,resolve,decision,conclusion,declare,announce,decrease,reduce,happyness,joy,gladness,demolish,destroy,denial,refusal,deny,refuse,denies,refuses,destination,goal,destiny,fate,colleague,coworker,small,tiiny,shout,yell,shouts,yells,speaks,talks,speaking,talking,clever,smart,present,gift,mother,mom,bunny,rabbit,garbage,trash,shuts,closes,shop,store,sees,looks,see,look,alike,same,chef,cook,crash,accident,raise,lift,stone,rock,stones,rocks,street,road,street,roads,near,close to,couch,sofa,father,dad,tired,sleepy,taxi,cab'.split(',')
    

    #words = 'avocado,avocados,carrot,carrots,celery,celeries,chick peas,cucumber,cucumbers,eggplant,eggplants,onion,onions,pumpkin,pumpkins,paotato,potatoes,tomato,tomatoes,vegetable,vegetables,zucchini,zucchinis'.split(',')
    words = 'ride to,rides to,crawl to,crawls to,run to,runs to,walk to,walks to,hurry to,hurries to,rush to,rushs to,stroll to,strolls to,drive to,drives to,fly to,flys to,swim to,swims to'.split(',')
    words_single = 'ride,rides,crawl,crawls,run,runs,jog,jogs,walk,walks,hurry,hurries,rush,rushs,stroll,strolls,drive,drives,swim,swims'.split(',')
    datahandler = data_manipulator.DataManipulator().load()
    datahandler.print_sents(words + words_single, 30)

    #name, repl1, repl2, repl_a = antonyms_other()
    #print('repl first')
    #for p, h, lbl in repl1:
    #    print(p, '--', h, '--', lbl)
    #print()
    #print('repl second')
    #for p, h, lbl in repl2:
    #    print(p, '--', h, '--', lbl)
    #print()
    #print('repl any')
    #for p, h, lbl in repl_a:
    #    print(p, '--', h, '--', lbl)
    #print()

def main():
    args = docopt("""Create a new dataset based on the given type.

    Usage:
        dataset_creator.py create <out_name>
        dataset_creator.py test 
        dataset_creator.py show -a <amount> (-w <words>)...
    """)


    if args['test']:
        test_out()
    elif args['show']:
        max_amount = int(args['<amount>'])
        words = args['<words>']
        datahandler = data_manipulator.DataManipulator().load()
        datahandler.print_sents(words, max_amount)
    else:
        out_name = args['<out_name>']
        all_fn = [
            countries,
            nationalities,
            colors,
            numbers,
            antonyms_adj_adv,
            antonyms_nn_vb,
            antonyms_other,
            synonyms
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
        

