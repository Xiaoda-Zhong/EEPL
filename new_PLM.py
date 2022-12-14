from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
special_tokens_dict = {'additional_special_tokens': [
  "arrests",
    "seized",
    "nominations",
    "bombing",
    "brawl",
    "loan",
    "sale",
    "launched",
    "set up",
    "appointment",
    "injure",
    "births",
    "extradite",
    "takeover",
    "pardoning",
    "fund",
    "took office",
    "fighting",
    "Compensation",
    "injury",
    "buys",
    "it",
    "summit",
    "hit",
    "action",
    "murder",
    "hanging",
    "Protests",
    "wrote",
    "close",
    "be",
    "letters",
    "re-election",
    "hanged",
    "terrorism",
    "formerly",
    "marries",
    "removal",
    "became",
    "acquiring",
    "retaliation",
    "taken",
    "bloodshed",
    "fine",
    "convicted",
    "sit-in",
    "assassination",
    "resignations",
    "cease",
    "fired",
    "slaughtered",
    "rallying",
    "appoint",
    "demonstrated",
    "held",
    "begins",
    "coming",
    "closing",
    "acquitting",
    "fight",
    "by-elections",
    "released",
    "arresting",
    "pays",
    "names",
    "wounds",
    "formed",
    "protest",
    "calling",
    "met",
    "flew",
    "air strikes",
    "creating",
    "hurts",
    "war",
    "election",
    "marriages",
    "payroll",
    "litigation",
    "acquire",
    "rallies",
    "raid",
    "hacked",
    "demonstrations",
    "complaint",
    "jihad",
    "reimbursing",
    "step down",
    "marriage",
    "buy",
    "tried",
    "landed",
    "battle",
    "sacked",
    "bought",
    "lawsuit",
    "together",
    "release",
    "wedding",
    "blowing up",
    "birth",
    "heading",
    "protests",
    "charging",
    "forum",
    "childbirth",
    "founding",
    "conflict",
    "convicts",
    "name",
    "suits",
    "charges",
    "convictions",
    "donation",
    "demonstration",
    "creation",
    "executed",
    "shot",
    "create",
    "injuring",
    "targeted",
    "appoints",
    "attack",
    "opened",
    "executions",
    "gunfire",
    "drafted",
    "warring",
    "gathered",
    "replace",
    "visited",
    "payment",
    "won",
    "hearing",
    "go",
    "votes",
    "leaving",
    "wed",
    "jailed",
    "payments",
    "cost",
    "tripping",
    "jailing",
    "strikes",
    "hear",
    "rally",
    "releases",
    "blow",
    "appeals",
    "trip",
    "acquittals",
    "blown",
    "topple",
    "injuries",
    "aggression",
    "vote",
    "ceased",
    "acquittal",
    "Married",
    "firing",
    "invasion",
    "quit",
    "hospitalised",
    "demonstrating",
    "raising money",
    "kill",
    "blew",
    "trials",
    "retired",
    "write",
    "pay",
    "protested",
    "selling",
    "compensation",
    "aquitted",
    "loaned",
    "elected",
    "electing",
    "sentenced",
    "resigning",
    "indictment",
    "reimburse",
    "birthing",
    "nominating",
    "wounding",
    "meet",
    "sentence",
    "took",
    "shed",
    "attacked",
    "bankruptcies",
    "travelling",
    "offensive",
    "call",
    "extradited",
    "bid",
    "comed",
    "wound",
    "named",
    "moved",
    "divorce",
    "talks",
    "buying out",
    "pardons",
    "aid package",
    "death",
    "trial",
    "give",
    "retire",
    "merge",
    "Retired",
    "accused",
    "paying",
    "come",
    "letter",
    "elect",
    "combat",
    "taking money",
    "shooting",
    "marrying",
    "leave",
    "hiring",
    "clashes",
    "ambush",
    "hearings",
    "arrives",
    "appealing",
    "indictments",
    "accusing",
    "jail",
    "fines",
    "talking",
    "this",
    "freed",
    "sues",
    "returned",
    "bombed",
    "convict",
    "transporting",
    "jails",
    "discussions",
    "wounded",
    "seize",
    "conference",
    "nominate",
    "wedded",
    "defected",
    "goes",
    "dying",
    "visits",
    "kills",
    "protesting",
    "breakdowns",
    "writes",
    "suit",
    "appointments",
    "sue",
    "called",
    "skirmishes",
    "compensating",
    "bankrupt",
    "came",
    "donations",
    "elections",
    "suicide",
    "divorcing",
    "seizing",
    "buyout",
    "divorces",
    "summits",
    "given",
    "attacking",
    "resignation",
    "nomination",
    "execute",
    "voting",
    "charged",
    "extraditing",
    "campaign",
    "crumble",
    "launching",
    "acquisitions",
    "pounded",
    "free",
    "visiting",
    "started",
    "went",
    "accuses",
    "sells",
    "retirement",
    "flown",
    "stepped down",
    "bombings",
    "gone",
    "visit",
    "hires",
    "assault",
    "sentences",
    "sentencing",
    "appeal",
    "hurt",
    "told",
    "injured",
    "firefight",
    "filled",
    "indicted",
    "born",
    "giving",
    "resign",
    "convicting",
    "bids",
    "conviction",
    "divorced",
    "get",
    "married",
    "raids",
    "moving",
    "deaths",
    "written",
    "ending",
    "compensate",
    "money",
    "eliminate",
    "loans",
    "Bankrupt",
    "bombardment",
    "suicides",
    "indict",
    "injures",
    "no_trigger",
    "appointing",
    "take over",
    "closed",
    "mergers",
    "inviting",
    "journeyed",
    "die",
    "comes",
    "naming",
    "demonstrate",
    "heard",
    "releasing",
    "killed",
    "charge",
    "blast",
    "Demonstrations",
    "fought",
    "transaction",
    "elects",
    "merged",
    "tripped",
    "arriving",
    "marched",
    "resigned",
    "execution",
    "opening",
    "previously",
    "travels",
    "wiped out",
    "War",
    "sued",
    "suing",
    "capture",
    "merger",
    "pardon",
    "meets",
    "extradition",
    "journey",
    "lawsuits",
    "guilty",
    "support",
    "move",
    "sell",
    "pardoned",
    "journeys",
    "arrested",
    "interview",
    "buying",
    "hire",
    "destruction",
    "retirements",
    "punched through",
    "killing",
    "weddings",
    "extraditions",
    "dip into",
    "telephone",
    "intifada",
    "rallied",
    "Founded",
    "dead",
    "founded",
    "meeting",
    "fly",
    "acquitted",
    "killings",
    "merging",
    "created",
    "appointed",
    "arrest",
    "destroyed",
    "calls",
    "writing",
    "attacks",
    "going",
    "dies",
    "nominated",
    "strike",
    "marry",
    "violence",
    "arrived",
    "wars",
    "convened",
    "voted",
    "fire",
    "acquit",
    "appealed",
    "paid",
    "nominates",
    "sold",
    "journeying",
    "accuse",
    "warfare",
    "trips",
    "Negotiations",
    "retiring",
    "moves",
    "fights",
    "former",
    "push",
    "died",
    "captured",
    "build",
    "arrive",
    "flying",
    "hired",
    "acquisition",
    "bankruptcy",
    "chopping off",
    "meetings"
]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model = AutoModel.from_pretrained("bert-base-uncased")
model.resize_token_embeddings(len(tokenizer))

tokenizer.save_pretrained("bert_new")
model.save_pretrained("bert_new")
