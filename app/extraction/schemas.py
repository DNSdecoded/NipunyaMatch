from pydantic import BaseModel, EmailStr, Field


class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: str | None = None  # ISO YYYY-MM where derivable
    end_date: str | None = None  # None means current
    description: str | None = None


class Education(BaseModel):
    institution: str
    degree: str | None = None
    field_of_study: str | None = None
    graduation_year: int | None = None


class CandidateExtraction(BaseModel):
    name: str | None = None
    email: EmailStr | None = None
    phone: str | None = None
    location: str | None = None
    linkedin_url: str | None = None
    skills: list[str] = Field(default_factory=list)  # verbatim from the resume, deduped
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    total_years_experience: float | None = None
    extraction_confidence: float = Field(default=0.5, ge=0, le=1)


class SkillMatch(BaseModel):
    skill: str
    present: bool
    evidence: str | None = None  # the resume phrase that proves it
    required: bool  # required vs nice-to-have in the JD


class CandidateAnalysis(BaseModel):
    llm_fit_score: int = Field(ge=0, le=100)
    skill_matches: list[SkillMatch]
    strengths: list[str] = Field(max_length=5)
    weaknesses: list[str] = Field(max_length=5)
    summary: str  # 2-3 sentences
    interview_questions: list[str] = Field(min_length=3, max_length=5)
    reasoning: str  # why this score, cited to resume content
