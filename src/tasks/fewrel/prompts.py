from ..utils_typing import Relation, dataclass


@dataclass
class OwnedBy(Relation):
	"""{OwnedBy}"""
	arg1: str
	arg2: str


@dataclass
class Publisher(Relation):
	"""{Publisher}"""
	arg1: str
	arg2: str


@dataclass
class Composer(Relation):
	"""{Composer}"""
	arg1: str
	arg2: str


@dataclass
class CastMember(Relation):
	"""{CastMember}"""
	arg1: str
	arg2: str


@dataclass
class LicensedToBroadcastTo(Relation):
	"""{LicensedToBroadcastTo}"""
	arg1: str
	arg2: str


@dataclass
class SaidToBeTheSameAs(Relation):
	"""{SaidToBeTheSameAs}"""
	arg1: str
	arg2: str


@dataclass
class Occupant(Relation):
	"""{Occupant}"""
	arg1: str
	arg2: str


@dataclass
class Sport(Relation):
	"""{Sport}"""
	arg1: str
	arg2: str


@dataclass
class MilitaryBranch(Relation):
	"""{MilitaryBranch}"""
	arg1: str
	arg2: str


@dataclass
class AwardReceived(Relation):
	"""{AwardReceived}"""
	arg1: str
	arg2: str


@dataclass
class Screenwriter(Relation):
	"""{Screenwriter}"""
	arg1: str
	arg2: str


@dataclass
class Constellation(Relation):
	"""{Constellation}"""
	arg1: str
	arg2: str


@dataclass
class MemberOfSportsTeam(Relation):
	"""{MemberOfSportsTeam}"""
	arg1: str
	arg2: str


@dataclass
class Director(Relation):
	"""{Director}"""
	arg1: str
	arg2: str


@dataclass
class Author(Relation):
	"""{Author}"""
	arg1: str
	arg2: str


@dataclass
class PlaceServedByTransportHub(Relation):
	"""{PlaceServedByTransportHub}"""
	arg1: str
	arg2: str


@dataclass
class AfterAWorkBy(Relation):
	"""{AfterAWorkBy}"""
	arg1: str
	arg2: str


@dataclass
class ProductOrMaterialProduced(Relation):
	"""{ProductOrMaterialProduced}"""
	arg1: str
	arg2: str


@dataclass
class NotableWork(Relation):
	"""{NotableWork}"""
	arg1: str
	arg2: str


@dataclass
class LocatedInOrNextToBodyOfWater(Relation):
	"""{LocatedInOrNextToBodyOfWater}"""
	arg1: str
	arg2: str


@dataclass
class Distributor(Relation):
	"""{Distributor}"""
	arg1: str
	arg2: str


@dataclass
class Instrument(Relation):
	"""{Instrument}"""
	arg1: str
	arg2: str


@dataclass
class Architect(Relation):
	"""{Architect}"""
	arg1: str
	arg2: str


@dataclass
class CauseOfDeath(Relation):
	"""{CauseOfDeath}"""
	arg1: str
	arg2: str


@dataclass
class Location(Relation):
	"""{Location}"""
	arg1: str
	arg2: str


@dataclass
class AppliesToJurisdiction(Relation):
	"""{AppliesToJurisdiction}"""
	arg1: str
	arg2: str


@dataclass
class Mother(Relation):
	"""{Mother}"""
	arg1: str
	arg2: str


@dataclass
class CountryOfCitizenship(Relation):
	"""{CountryOfCitizenship}"""
	arg1: str
	arg2: str


@dataclass
class Spouse(Relation):
	"""{Spouse}"""
	arg1: str
	arg2: str


@dataclass
class Father(Relation):
	"""{Father}"""
	arg1: str
	arg2: str


@dataclass
class Manufacturer(Relation):
	"""{Manufacturer}"""
	arg1: str
	arg2: str


@dataclass
class Participant(Relation):
	"""{Participant}"""
	arg1: str
	arg2: str


@dataclass
class Winner(Relation):
	"""{Winner}"""
	arg1: str
	arg2: str


@dataclass
class FollowedBy(Relation):
	"""{FollowedBy}"""
	arg1: str
	arg2: str


@dataclass
class Follows(Relation):
	"""{Follows}"""
	arg1: str
	arg2: str


@dataclass
class ContainsAdministrativeTerritorialEntity(Relation):
	"""{ContainsAdministrativeTerritorialEntity}"""
	arg1: str
	arg2: str


@dataclass
class ParticipantOf(Relation):
	"""{ParticipantOf}"""
	arg1: str
	arg2: str


@dataclass
class Country(Relation):
	"""{Country}"""
	arg1: str
	arg2: str


@dataclass
class SuccessfulCandidate(Relation):
	"""{SuccessfulCandidate}"""
	arg1: str
	arg2: str


@dataclass
class OriginalNetwork(Relation):
	"""{OriginalNetwork}"""
	arg1: str
	arg2: str


@dataclass
class LocationOfFormation(Relation):
	"""{LocationOfFormation}"""
	arg1: str
	arg2: str


@dataclass
class ParentOrganization(Relation):
	"""{ParentOrganization}"""
	arg1: str
	arg2: str


@dataclass
class DirectorOfPhotography(Relation):
	"""{DirectorOfPhotography}"""
	arg1: str
	arg2: str


@dataclass
class RecordLabel(Relation):
	"""{RecordLabel}"""
	arg1: str
	arg2: str


@dataclass
class ParticipatingTeam(Relation):
	"""{ParticipatingTeam}"""
	arg1: str
	arg2: str


@dataclass
class Employer(Relation):
	"""{Employer}"""
	arg1: str
	arg2: str


@dataclass
class TaxonRank(Relation):
	"""{TaxonRank}"""
	arg1: str
	arg2: str


@dataclass
class Occupation(Relation):
	"""{Occupation}"""
	arg1: str
	arg2: str


@dataclass
class FieldOfWork(Relation):
	"""{FieldOfWork}"""
	arg1: str
	arg2: str


@dataclass
class MemberOfPoliticalParty(Relation):
	"""{MemberOfPoliticalParty}"""
	arg1: str
	arg2: str


@dataclass
class MemberOf(Relation):
	"""{MemberOf}"""
	arg1: str
	arg2: str


@dataclass
class StockExchange(Relation):
	"""{StockExchange}"""
	arg1: str
	arg2: str


@dataclass
class PositionHeld(Relation):
	"""{PositionHeld}"""
	arg1: str
	arg2: str


@dataclass
class InstanceOf(Relation):
	"""{InstanceOf}"""
	arg1: str
	arg2: str


@dataclass
class Platform(Relation):
	"""{Platform}"""
	arg1: str
	arg2: str


@dataclass
class LanguageOfWorkOrName(Relation):
	"""{LanguageOfWorkOrName}"""
	arg1: str
	arg2: str


@dataclass
class LocatedOnTerrainFeature(Relation):
	"""{LocatedOnTerrainFeature}"""
	arg1: str
	arg2: str


@dataclass
class CountryOfOrigin(Relation):
	"""{CountryOfOrigin}"""
	arg1: str
	arg2: str


@dataclass
class HeadquartersLocation(Relation):
	"""{HeadquartersLocation}"""
	arg1: str
	arg2: str


@dataclass
class Religion(Relation):
	"""{Religion}"""
	arg1: str
	arg2: str


@dataclass
class HeritageDesignation(Relation):
	"""{HeritageDesignation}"""
	arg1: str
	arg2: str


@dataclass
class WorkLocation(Relation):
	"""{WorkLocation}"""
	arg1: str
	arg2: str


@dataclass
class Movement(Relation):
	"""{Movement}"""
	arg1: str
	arg2: str


@dataclass
class Genre(Relation):
	"""{Genre}"""
	arg1: str
	arg2: str


@dataclass
class Operator(Relation):
	"""{Operator}"""
	arg1: str
	arg2: str


@dataclass
class LocatedInTheAdministrativeTerritorialEntity(Relation):
	"""{LocatedInTheAdministrativeTerritorialEntity}"""
	arg1: str
	arg2: str


@dataclass
class HasPart(Relation):
	"""{HasPart}"""
	arg1: str
	arg2: str


@dataclass
class NominatedFor(Relation):
	"""{NominatedFor}"""
	arg1: str
	arg2: str


@dataclass
class MainSubject(Relation):
	"""{MainSubject}"""
	arg1: str
	arg2: str


@dataclass
class Subsidiary(Relation):
	"""{Subsidiary}"""
	arg1: str
	arg2: str


@dataclass
class Industry(Relation):
	"""{Industry}"""
	arg1: str
	arg2: str


@dataclass
class MouthOfTheWatercourse(Relation):
	"""{MouthOfTheWatercourse}"""
	arg1: str
	arg2: str


@dataclass
class Sibling(Relation):
	"""{Sibling}"""
	arg1: str
	arg2: str


@dataclass
class MaterialUsed(Relation):
	"""{MaterialUsed}"""
	arg1: str
	arg2: str


@dataclass
class Characters(Relation):
	"""{Characters}"""
	arg1: str
	arg2: str


@dataclass
class SportsSeasonOfLeagueOrCompetition(Relation):
	"""{SportsSeasonOfLeagueOrCompetition}"""
	arg1: str
	arg2: str


@dataclass
class Developer(Relation):
	"""{Developer}"""
	arg1: str
	arg2: str


@dataclass
class ProductionCompany(Relation):
	"""{ProductionCompany}"""
	arg1: str
	arg2: str


@dataclass
class ParentTaxon(Relation):
	"""{ParentTaxon}"""
	arg1: str
	arg2: str


@dataclass
class EthnicGroup(Relation):
	"""{EthnicGroup}"""
	arg1: str
	arg2: str


@dataclass
class Performer(Relation):
	"""{Performer}"""
	arg1: str
	arg2: str


@dataclass
class Crosses(Relation):
	"""{Crosses}"""
	arg1: str
	arg2: str


@dataclass
class CompetitionClass(Relation):
	"""{CompetitionClass}"""
	arg1: str
	arg2: str


@dataclass
class Tributary(Relation):
	"""{Tributary}"""
	arg1: str
	arg2: str


@dataclass
class MountainRange(Relation):
	"""{MountainRange}"""
	arg1: str
	arg2: str


@dataclass
class Child(Relation):
	"""{Child}"""
	arg1: str
	arg2: str


@dataclass
class PositionPlayedOnTeamOrSpeciality(Relation):
	"""{PositionPlayedOnTeamOrSpeciality}"""
	arg1: str
	arg2: str


@dataclass
class VoiceType(Relation):
	"""{VoiceType}"""
	arg1: str
	arg2: str


@dataclass
class MilitaryRank(Relation):
	"""{MilitaryRank}"""
	arg1: str
	arg2: str


@dataclass
class PresentInWork(Relation):
	"""{PresentInWork}"""
	arg1: str
	arg2: str


@dataclass
class Residence(Relation):
	"""{Residence}"""
	arg1: str
	arg2: str


@dataclass
class PartOf(Relation):
	"""{PartOf}"""
	arg1: str
	arg2: str


@dataclass
class OriginalLanguageOfFilmOrTvShow(Relation):
	"""{OriginalLanguageOfFilmOrTvShow}"""
	arg1: str
	arg2: str


@dataclass
class OperatingSystem(Relation):
	"""{OperatingSystem}"""
	arg1: str
	arg2: str


@dataclass
class Producer(Relation):
	"""{Producer}"""
	arg1: str
	arg2: str


@dataclass
class HeadOfGovernment(Relation):
	"""{HeadOfGovernment}"""
	arg1: str
	arg2: str


@dataclass
class League(Relation):
	"""{League}"""
	arg1: str
	arg2: str



COARSE_RELATION_DEFINITIONS: list = [
	OwnedBy,
	Publisher,
	Composer,
	CastMember,
	LicensedToBroadcastTo,
	SaidToBeTheSameAs,
	Occupant,
	Sport,
	MilitaryBranch,
	AwardReceived,
	Screenwriter,
	Constellation,
	MemberOfSportsTeam,
	Director,
	Author,
	PlaceServedByTransportHub,
	AfterAWorkBy,
	ProductOrMaterialProduced,
	NotableWork,
	LocatedInOrNextToBodyOfWater,
	Distributor,
	Instrument,
	Architect,
	CauseOfDeath,
	Location,
	AppliesToJurisdiction,
	Mother,
	CountryOfCitizenship,
	Spouse,
	Father,
	Manufacturer,
	Participant,
	Winner,
	FollowedBy,
	Follows,
	ContainsAdministrativeTerritorialEntity,
	ParticipantOf,
	Country,
	SuccessfulCandidate,
	OriginalNetwork,
	LocationOfFormation,
	ParentOrganization,
	DirectorOfPhotography,
	RecordLabel,
	ParticipatingTeam,
	Employer,
	TaxonRank,
	Occupation,
	FieldOfWork,
	MemberOfPoliticalParty,
	MemberOf,
	StockExchange,
	PositionHeld,
	InstanceOf,
	Platform,
	LanguageOfWorkOrName,
	LocatedOnTerrainFeature,
	CountryOfOrigin,
	HeadquartersLocation,
	Religion,
	HeritageDesignation,
	WorkLocation,
	Movement,
	Genre,
	Operator,
	LocatedInTheAdministrativeTerritorialEntity,
	HasPart,
	NominatedFor,
	MainSubject,
	Subsidiary,
	Industry,
	MouthOfTheWatercourse,
	Sibling,
	MaterialUsed,
	Characters,
	SportsSeasonOfLeagueOrCompetition,
	Developer,
	ProductionCompany,
	ParentTaxon,
	EthnicGroup,
	Performer,
	Crosses,
	CompetitionClass,
	Tributary,
	MountainRange,
	Child,
	PositionPlayedOnTeamOrSpeciality,
	VoiceType,
	MilitaryRank,
	PresentInWork,
	Residence,
	PartOf,
	OriginalLanguageOfFilmOrTvShow,
	OperatingSystem,
	Producer,
	HeadOfGovernment,
	League,
]

REL2CLSMAPPING: dict = {
	"owned by": OwnedBy,
	"publisher": Publisher,
	"composer": Composer,
	"cast member": CastMember,
	"licensed to broadcast to": LicensedToBroadcastTo,
	"said to be the same as": SaidToBeTheSameAs,
	"occupant": Occupant,
	"sport": Sport,
	"military branch": MilitaryBranch,
	"award received": AwardReceived,
	"screenwriter": Screenwriter,
	"constellation": Constellation,
	"member of sports team": MemberOfSportsTeam,
	"director": Director,
	"author": Author,
	"place served by transport hub": PlaceServedByTransportHub,
	"after a work by": AfterAWorkBy,
	"product or material produced": ProductOrMaterialProduced,
	"notable work": NotableWork,
	"located in or next to body of water": LocatedInOrNextToBodyOfWater,
	"distributor": Distributor,
	"instrument": Instrument,
	"architect": Architect,
	"cause of death": CauseOfDeath,
	"location": Location,
	"applies to jurisdiction": AppliesToJurisdiction,
	"mother": Mother,
	"country of citizenship": CountryOfCitizenship,
	"spouse": Spouse,
	"father": Father,
	"manufacturer": Manufacturer,
	"participant": Participant,
	"winner": Winner,
	"followed by": FollowedBy,
	"follows": Follows,
	"contains administrative territorial entity": ContainsAdministrativeTerritorialEntity,
	"participant of": ParticipantOf,
	"country": Country,
	"successful candidate": SuccessfulCandidate,
	"original network": OriginalNetwork,
	"location of formation": LocationOfFormation,
	"parent organization": ParentOrganization,
	"director of photography": DirectorOfPhotography,
	"record label": RecordLabel,
	"participating team": ParticipatingTeam,
	"employer": Employer,
	"taxon rank": TaxonRank,
	"occupation": Occupation,
	"field of work": FieldOfWork,
	"member of political party": MemberOfPoliticalParty,
	"member of": MemberOf,
	"stock exchange": StockExchange,
	"position held": PositionHeld,
	"instance of": InstanceOf,
	"platform": Platform,
	"language of work or name": LanguageOfWorkOrName,
	"located on terrain feature": LocatedOnTerrainFeature,
	"country of origin": CountryOfOrigin,
	"headquarters location": HeadquartersLocation,
	"religion": Religion,
	"heritage designation": HeritageDesignation,
	"work location": WorkLocation,
	"movement": Movement,
	"genre": Genre,
	"operator": Operator,
	"located in the administrative territorial entity": LocatedInTheAdministrativeTerritorialEntity,
	"has part": HasPart,
	"nominated for": NominatedFor,
	"main subject": MainSubject,
	"subsidiary": Subsidiary,
	"industry": Industry,
	"mouth of the watercourse": MouthOfTheWatercourse,
	"sibling": Sibling,
	"material used": MaterialUsed,
	"characters": Characters,
	"sports season of league or competition": SportsSeasonOfLeagueOrCompetition,
	"developer": Developer,
	"production company": ProductionCompany,
	"parent taxon": ParentTaxon,
	"ethnic group": EthnicGroup,
	"performer": Performer,
	"crosses": Crosses,
	"competition class": CompetitionClass,
	"tributary": Tributary,
	"mountain range": MountainRange,
	"child": Child,
	"position played on team / speciality": PositionPlayedOnTeamOrSpeciality,
	"voice type": VoiceType,
	"military rank": MilitaryRank,
	"present in work": PresentInWork,
	"residence": Residence,
	"part of": PartOf,
	"original language of film or TV show": OriginalLanguageOfFilmOrTvShow,
	"operating system": OperatingSystem,
	"producer": Producer,
	"head of government": HeadOfGovernment,
	"league": League,
}