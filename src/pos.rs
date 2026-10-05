/// A part of speech, in the OpenCorpora tag set with verbs split by mood.
///
/// The order matches the label order in the `.vrb` file; loading checks it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pos {
    /// Существительное.
    Noun,
    /// Полное прилагательное.
    Adjf,
    /// Краткое прилагательное.
    Adjs,
    /// Компаратив.
    Comp,
    /// Глагол в изъявительном наклонении.
    Verb,
    /// Глагол в повелительном наклонении.
    Impr,
    /// Инфинитив.
    Infn,
    /// Полное причастие.
    Prtf,
    /// Краткое причастие.
    Prts,
    /// Деепричастие.
    Grnd,
    /// Числительное.
    Numr,
    /// Наречие.
    Advb,
    /// Местоимение-существительное.
    Npro,
    /// Предикатив.
    Pred,
    /// Предлог.
    Prep,
    /// Союз.
    Conj,
    /// Частица.
    Prcl,
    /// Междометие.
    Intj,
}

impl Pos {
    pub const ALL: [Pos; 18] = [
        Pos::Noun,
        Pos::Adjf,
        Pos::Adjs,
        Pos::Comp,
        Pos::Verb,
        Pos::Impr,
        Pos::Infn,
        Pos::Prtf,
        Pos::Prts,
        Pos::Grnd,
        Pos::Numr,
        Pos::Advb,
        Pos::Npro,
        Pos::Pred,
        Pos::Prep,
        Pos::Conj,
        Pos::Prcl,
        Pos::Intj,
    ];

    /// The OpenCorpora tag, e.g. `"INFN"`; `"IMPR"` for imperatives.
    pub fn tag(self) -> &'static str {
        match self {
            Pos::Noun => "NOUN",
            Pos::Adjf => "ADJF",
            Pos::Adjs => "ADJS",
            Pos::Comp => "COMP",
            Pos::Verb => "VERB",
            Pos::Impr => "IMPR",
            Pos::Infn => "INFN",
            Pos::Prtf => "PRTF",
            Pos::Prts => "PRTS",
            Pos::Grnd => "GRND",
            Pos::Numr => "NUMR",
            Pos::Advb => "ADVB",
            Pos::Npro => "NPRO",
            Pos::Pred => "PRED",
            Pos::Prep => "PREP",
            Pos::Conj => "CONJ",
            Pos::Prcl => "PRCL",
            Pos::Intj => "INTJ",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Pos> {
        Pos::ALL.into_iter().find(|p| p.tag() == tag)
    }

    /// An infinitive or an imperative: a phrase starting with it names an
    /// action («позвонить маме», «купи молоко»).
    pub fn is_action(self) -> bool {
        matches!(self, Pos::Infn | Pos::Impr)
    }
}
