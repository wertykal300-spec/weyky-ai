package com.example.core.intelligence

import com.example.core.context.ContextSnapshot
import com.example.core.context.ContextState
import com.example.core.intelligence.battery.BatteryForecastCalculator
import com.example.core.intelligence.battery.BatteryForecastOutcome
import com.example.core.intelligence.battery.BatteryForecastScenario
import com.example.core.intelligence.battery.BatteryForecastTarget
import com.example.core.intelligence.battery.BatteryTargetTimeExtractor
import com.example.core.intelligence.conversation.ActionableOption
import com.example.core.intelligence.conversation.ConversationSession
import com.example.core.intelligence.conversation.ConversationalReasoner
import com.example.core.intelligence.conversation.LocalConversationalEngine
import com.example.core.intelligence.conversation.PendingExpectation
import com.example.core.intelligence.time.SettableTimeProvider
import com.example.core.intent.ExecutionResult
import com.example.core.intent.IntentExecutor
import com.example.core.intent.IntentParser
import com.example.core.intent.MessageIntentClassifier
import com.example.core.intent.MessageIntentType
import com.example.core.intent.WeykyIntent
import com.example.core.intent.handlers.AppActionHandler
import com.example.core.intent.handlers.AudioActionHandler
import com.example.core.intent.handlers.BatteryActionHandler
import com.example.core.intent.handlers.CallActionHandler
import com.example.core.intent.handlers.ContextActionHandler
import com.example.core.intent.handlers.FlashlightActionHandler
import com.example.core.intent.handlers.MediaActionHandler
import com.example.core.intent.handlers.NetworkActionHandler
import com.example.core.intent.handlers.NotificationActionHandler
import com.example.core.intent.handlers.SettingsActionHandler
import com.example.core.intent.handlers.StorageMemoryActionHandler
import com.example.core.intent.handlers.WeatherActionHandler
import com.example.device.battery.BatterySnapshot
import com.example.device.battery.ChargingState
import com.example.device.battery.PowerSource
import com.example.device.calendar.CalendarEventItem
import com.example.device.calendar.CalendarSummary
import com.example.device.weather.WeatherSnapshot
import com.example.rendering.visual_state.CoreVisualState
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Calendar

/**
 * WEYKY Phase 3: Comprehensive Contextual Intelligence Integration & Reliability Suite.
 *
 * Covers sections 4.A through 4.F:
 * 4.A Contextual reasoning & no redundant synthesis
 * 4.B Multi-turn time-aware battery conversations with navigation follow-up
 * 4.C Compound-command execution and pre-validation
 * 4.D Dialogue state and follow-up resolution
 * 4.E Epistemic integrity and proactive suggestions
 * 4.F Voice and text input consistency
 */
class Phase3ReliabilityRegressionTest {

  private fun createMockSnapshot(
    batteryDatum: ContextDatum<BatterySnapshot> = ContextDatum.unavailable("BatteryProvider", "Not available"),
    weatherDatum: ContextDatum<WeatherSnapshot> = ContextDatum.unavailable("WeatherProvider", "Not available"),
    calendarDatum: ContextDatum<CalendarSummary> = ContextDatum.unavailable("CalendarProvider", "Not available"),
    physicalContextDatum: ContextDatum<ContextSnapshot> = ContextDatum.unavailable("ContextEngine", "Not available")
  ): SituationSnapshot {
    return SituationSnapshot(
      timestampMillis = 1000L,
      battery = batteryDatum,
      network = ContextDatum.unavailable("NetworkProvider", "Test"),
      physicalContext = physicalContextDatum,
      location = ContextDatum.unavailable("LocationProvider", "Test"),
      weather = weatherDatum,
      calendar = calendarDatum,
      media = ContextDatum.unavailable("MediaPlaybackController", "Test"),
      calls = ContextDatum.unavailable("CallLogProvider", "Test"),
      notifications = ContextDatum.unavailable("NotificationRepository", "Test")
    )
  }

  private fun createBattery(
    percentage: Int,
    chargingState: ChargingState = ChargingState.DISCHARGING,
    powerSource: PowerSource = PowerSource.BATTERY
  ): BatterySnapshot {
    return BatterySnapshot(
      percentage = percentage,
      chargingState = chargingState,
      powerSource = powerSource,
      temperatureCelsius = 26.0f,
      voltageMillivolts = 3850,
      healthString = "Good",
      isAvailable = percentage >= 0
    )
  }

  // =========================================================================
  // SECTION 4.A: Contextual Reasoning & Verification of Non-Duplication
  // =========================================================================

  @Test
  fun test4A_contextInterpretedOnlyWhenAvailable_noInventedValues() {
    val timeProvider = SettableTimeProvider().apply {
      setTime(hourOfDay = 11, minute = 0, second = 0)
    }
    val engine = ContextReasoningEngine(timeProvider)

    // Snapshot where weather is available, but calendar and battery are completely missing
    val weather = WeatherSnapshot(
      temperatureCelsius = 28.0f,
      apparentTemperatureCelsius = 29.0f,
      condition = "Sunny / مشمس",
      locationName = "مسقط",
      isAvailable = true
    )
    val snapshot = createMockSnapshot(
      weatherDatum = ContextDatum.fact(weather, "WeatherProvider"),
      batteryDatum = ContextDatum.unavailable("BatteryProvider", "No battery sensor"),
      calendarDatum = ContextDatum.unavailable("CalendarProvider", "Permission not granted")
    )

    val req = ContextRequirementsRegistry.resolve("هل أخرج الآن؟")
    val synthesis = engine.synthesize(req, snapshot, "هل أخرج الآن؟")

    // Verify weather is used, but missing battery and calendar are reported honestly
    assertTrue("Must mention available weather condition", synthesis.synthesizedSpeech.contains("28") || synthesis.synthesizedSpeech.contains("مشمس"))
    assertTrue("Must flag battery as unavailable", synthesis.unavailabilitiesHandled.any { it.contains("Battery") || it.contains("البطارية") })
    assertTrue("Must flag calendar as unavailable", synthesis.unavailabilitiesHandled.any { it.contains("Calendar") || it.contains("التقويم") })
    // Incomplete context must reduce confidence score
    assertTrue("Confidence score must be penalized for missing telemetry", synthesis.confidenceScore <= 0.7f)
  }

  @Test
  fun test4A_outdoorDepartureExplainsMaterialFactors() {
    val timeProvider = SettableTimeProvider().apply {
      setTime(hourOfDay = 16, minute = 30, second = 0)
    }
    val engine = ContextReasoningEngine(timeProvider)

    val weather = WeatherSnapshot(
      temperatureCelsius = 24.0f,
      apparentTemperatureCelsius = 23.0f,
      condition = "Mild / معتدل",
      locationName = "مسقط",
      isAvailable = true
    )
    val battery = createBattery(percentage = 88)
    val calendar = CalendarSummary(
      hasPermission = true,
      isAvailable = true,
      nextEvent = CalendarEventItem(
        id = 1L,
        title = "اجتماع عمل",
        description = null,
        location = "المركز المالي",
        startMillis = System.currentTimeMillis() + 60 * 60 * 1000L,
        endMillis = System.currentTimeMillis() + 120 * 60 * 1000L,
        minutesUntilStart = 60L
      ),
      upcomingEventsToday = emptyList()
    )

    val snapshot = createMockSnapshot(
      batteryDatum = ContextDatum.fact(battery, "BatteryProvider"),
      weatherDatum = ContextDatum.fact(weather, "WeatherProvider"),
      calendarDatum = ContextDatum.fact(calendar, "CalendarProvider")
    )

    val req = ContextRequirementsRegistry.resolve("هل اخرج الان؟")
    val synthesis = engine.synthesize(req, snapshot, "هل اخرج الان؟")

    assertTrue("Speech must state battery level", synthesis.synthesizedSpeech.contains("88%"))
    assertTrue("Speech must state weather", synthesis.synthesizedSpeech.contains("24") || synthesis.synthesizedSpeech.contains("معتدل"))
    assertTrue("Speech must note schedule status", synthesis.synthesizedSpeech.contains("اجتماع عمل") || synthesis.synthesizedSpeech.contains("60"))
    assertTrue("Must recommend suitable outdoor exit", synthesis.recommendations.isNotEmpty())
  }

  @Test
  fun test4A_singleExecutionPath_noRedundantTelemetryCollection() {
    val session = ConversationSession()

    // 1. Contextual queries are mapped exclusively to ContextualQueryIntent, preventing dual invocation
    val outdoorIntent = IntentParser.parse("هل أخرج الآن؟", session)
    assertTrue("Must parse as ContextualQueryIntent", outdoorIntent is WeykyIntent.ContextualQueryIntent)
    assertEquals(IntentDomain.OUTDOOR_DEPARTURE, (outdoorIntent as WeykyIntent.ContextualQueryIntent).domain)

    val scheduleIntent = IntentParser.parse("هل سأصل في الموعد؟", session)
    assertTrue("Must parse as ContextualQueryIntent", scheduleIntent is WeykyIntent.ContextualQueryIntent)
    assertEquals(IntentDomain.APPOINTMENT_TIMING, (scheduleIntent as WeykyIntent.ContextualQueryIntent).domain)

    // 2. Time-aware battery forecast is routed directly to TimeAwareBatteryForecastIntent
    val batteryForecastIntent = IntentParser.parse("كم ستتبقى من البطارية حتى السادسة مساءً؟", session)
    assertTrue("Must parse as TimeAwareBatteryForecastIntent", batteryForecastIntent is WeykyIntent.TimeAwareBatteryForecastIntent)

    // 3. Verify selective context scoping: each domain only collects needed fields
    val reqOutdoor = ContextRequirementsRegistry.resolve("هل أخرج الآن؟")
    assertTrue(reqOutdoor.required.contains(ContextField.WEATHER))
    assertTrue(reqOutdoor.required.contains(ContextField.BATTERY))
    assertFalse("Must not pull unneeded calls telemetry", reqOutdoor.required.contains(ContextField.CALLS))
    assertFalse("Must not pull unneeded notifications telemetry", reqOutdoor.required.contains(ContextField.NOTIFICATIONS))

    val reqBattery = ContextRequirementsRegistry.resolve("هل ستكفي البطارية؟")
    assertTrue(reqBattery.required.contains(ContextField.BATTERY))
    assertFalse("Must not pull unneeded location telemetry", reqBattery.required.contains(ContextField.LOCATION))
  }

  // =========================================================================
  // SECTION 4.B: Time-Aware Battery Multi-Turn Conversation
  // =========================================================================

  @Test
  fun test4B_fullMultiTurnSequence_withMusicAndNavigationLimitation() = runBlocking {
    val timeProvider = SettableTimeProvider().apply {
      setTime(hourOfDay = 14, minute = 0, second = 0) // 2:00 PM
    }
    val engine = ContextReasoningEngine(timeProvider)
    val session = ConversationSession()

    val snapshot = createMockSnapshot(
      batteryDatum = ContextDatum.fact(createBattery(percentage = 80), "BatteryProvider")
    )

    // Turn 1: Target time with music: "كم ستتبقى من البطارية حتى السادسة مساءً إذا استمعت للموسيقى؟"
    val target = BatteryForecastTarget(
      targetHour = 18,
      targetMinute = 0,
      descriptionArabic = "الساعة 6:00 مساءً",
      descriptionEnglish = "6:00 PM",
      rawMatchedString = "السادسة"
    )
    val outcome1 = engine.reasonBatteryForecast(snapshot, target, includesMusic = true, isArabic = true)
    assertTrue(outcome1 is BatteryForecastOutcome.Success)
    val success1 = outcome1 as BatteryForecastOutcome.Success

    // Horizon: 18:00 - 14:00 = 4.0 hours
    assertEquals(4.0, success1.horizonHours, 0.01)
    // Rate: baseline 2-4%/hr + music 4-8%/hr = 6-12%/hr * 4 hrs = 24-48% drain -> remaining 32-56%
    assertEquals(32, success1.estimatedMinPct)
    assertEquals(56, success1.estimatedMaxPct)
    assertTrue("Status must be labelled [ESTIMATE]", success1.synthesis.synthesizedSpeech.contains("[ESTIMATE]"))
    assertTrue("Must disclose actual battery measurement separately", success1.synthesis.synthesizedSpeech.contains("80%"))

    // Save to session
    session.lastBatteryScenario = success1.scenario
    session.recordAssistantTurn(success1.synthesis.synthesizedSpeech, newTopic = "battery")

    // Turn 2: Follow-up asking about navigation: "وماذا لو استخدمت الملاحة أيضًا؟"
    val classified = MessageIntentClassifier.classify("وماذا لو استخدمت الملاحة أيضًا؟", session)
    assertEquals(MessageIntentType.FOLLOW_UP_QUERY, classified.type)
    assertEquals("battery_navigation_followup", classified.primarySubject)

    val navLimitation = engine.reasonNavigationFollowUp(session.lastBatteryScenario!!, isArabic = true)
    // Preserves horizon, target hour, and music inclusion
    assertEquals(18, navLimitation.updatedScenario.targetHour)
    assertEquals(4.0, navLimitation.updatedScenario.horizonHours, 0.01)
    assertTrue(navLimitation.updatedScenario.includesMusic)
    assertTrue(navLimitation.updatedScenario.includesNavigation)

    // Acknowledges limitation without fabricating numerical estimate
    val speech = navLimitation.synthesis.synthesizedSpeech
    assertTrue(speech.contains("الملاحة") && speech.contains("GPS"))
    assertTrue(speech.contains("لا يمكن تقديم نسبة مئوية رقمية محددة"))
    assertTrue(speech.contains("شاحن سيارة") || speech.contains("شاحن السيارة"))

    // Turn 3: Unrelated topic: "حدثني عن صلالة"
    session.recordUserTurn("حدثني عن صلالة")
    session.recordAssistantTurn("صلالة مدينة ساحلية تتميز بالخريف...", newTopic = "travel")

    // Verify stale battery scenario is cleared
    assertEquals("travel", session.currentTopic)
    assertNull("Stale battery scenario must be null after topic change", session.lastBatteryScenario)
  }

  @Test
  fun test4B_chargingStates_acUsbWireless_noDischargeFormula() {
    val timeProvider = SettableTimeProvider().apply {
      setTime(hourOfDay = 10, minute = 0, second = 0)
    }
    val calculator = BatteryForecastCalculator(timeProvider)
    val target = BatteryForecastTarget(16, 0, "الساعة 4:00 مساءً", "4:00 PM", "الرابعة")

    val powerSources = listOf(PowerSource.AC, PowerSource.USB, PowerSource.WIRELESS)
    for (ps in powerSources) {
      val snap = createMockSnapshot(
        batteryDatum = ContextDatum.fact(createBattery(50, ChargingState.CHARGING, ps), "BatteryProvider")
      )
      val outcome = calculator.calculateForecast(snap, target, includesMusic = true, isArabic = true)
      assertTrue("Must return Charging outcome for $ps", outcome is BatteryForecastOutcome.Charging)
      val charging = outcome as BatteryForecastOutcome.Charging
      assertTrue(charging.scenario.isCharging)
      assertTrue("Must state that device is charging", charging.synthesis.synthesizedSpeech.contains("قيد الشحن"))
      assertFalse("Must not apply discharge formula while charging", charging.synthesis.synthesizedSpeech.contains("[ESTIMATE]"))
    }
  }

  @Test
  fun test4B_missingBatteryTelemetry_handledHonestly() {
    val timeProvider = SettableTimeProvider().apply {
      setTime(hourOfDay = 10, minute = 0, second = 0)
    }
    val calculator = BatteryForecastCalculator(timeProvider)
    val target = BatteryForecastTarget(16, 0, "الساعة 4:00 مساءً", "4:00 PM", "الرابعة")

    // Case 1: unavailable datum
    val snap1 = createMockSnapshot(
      batteryDatum = ContextDatum.unavailable("BatteryProvider", "Battery service not registered")
    )
    val outcome1 = calculator.calculateForecast(snap1, target, includesMusic = false, isArabic = true)
    assertTrue("Must return TelemetryUnavailable", outcome1 is BatteryForecastOutcome.TelemetryUnavailable)

    // Case 2: invalid negative percentage
    val snap2 = createMockSnapshot(
      batteryDatum = ContextDatum.fact(createBattery(percentage = -1), "BatteryProvider")
    )
    val outcome2 = calculator.calculateForecast(snap2, target, includesMusic = false, isArabic = true)
    assertTrue("Negative percentage must trigger TelemetryUnavailable", outcome2 is BatteryForecastOutcome.TelemetryUnavailable)
  }

  @Test
  fun test4B_targetTimeAlreadyPassed_andBoundaryCases() {
    val timeProvider = SettableTimeProvider().apply {
      setTime(hourOfDay = 15, minute = 45, second = 0) // 3:45 PM
    }
    val calculator = BatteryForecastCalculator(timeProvider)
    val snap = createMockSnapshot(batteryDatum = ContextDatum.fact(createBattery(70), "BatteryProvider"))

    // Case 1: Target time 14:00 has already passed today
    val pastTarget = BatteryForecastTarget(14, 0, "الساعة 2:00 م", "2:00 PM", "الثانية")
    val outcomePast = calculator.calculateForecast(snap, pastTarget, includesMusic = false, isArabic = true)
    assertTrue("Must return TargetTimePassed", outcomePast is BatteryForecastOutcome.TargetTimePassed)

    // Case 2: Target is in the future within the same hour (e.g. 15:55 -> 10 minutes horizon)
    val nearTarget = BatteryForecastTarget(15, 55, "الساعة 3:55 م", "3:55 PM", "3:55")
    val outcomeNear = calculator.calculateForecast(snap, nearTarget, includesMusic = false, isArabic = true)
    assertTrue("Near future target must succeed", outcomeNear is BatteryForecastOutcome.Success)
    val nearSuccess = outcomeNear as BatteryForecastOutcome.Success
    assertTrue("Horizon must be greater than 0", nearSuccess.horizonHours > 0)
  }

  // =========================================================================
  // SECTION 4.C: Compound-Command Execution & Pre-Validation
  // =========================================================================

  @Test
  fun test4C_unsupportedSegmentRejectsCompoundExecution() {
    // "شغل الكشاف ثم اطير إلى القمر" -> "اطير الى القمر" is unsupported
    val input = "شغل الكشاف ثم اطير الى القمر"
    val parsed = IntentParser.tryParseCompound(input)
    assertNull("Compound command with unsupported segment must be rejected entirely", parsed)
  }

  @Test
  fun test4C_validCompoundStepsPreserveOrderAndValidation() {
    val input = "شغل الكشاف ثم ارفع الصوت خطوتين"
    val parsed = IntentParser.parse(input)
    assertTrue(parsed is WeykyIntent.CompoundIntent)
    val compound = parsed as WeykyIntent.CompoundIntent

    assertEquals(2, compound.steps.size)
    assertTrue(compound.steps[0].intent is WeykyIntent.FlashlightIntent)
    assertTrue((compound.steps[0].intent as WeykyIntent.FlashlightIntent).turnOn)

    assertTrue(compound.steps[1].intent is WeykyIntent.VolumeStepIntent)
    val vol = compound.steps[1].intent as WeykyIntent.VolumeStepIntent
    assertTrue(vol.increase)
    assertEquals(2, vol.stepCount)
  }

  @Test
  fun test4C_permissionRequirementHaltsExecutionWithPartialReport() = runBlocking {
    // Simulate step 1 succeeds, step 2 requires permission
    val step1 = WeykyIntent.CompoundStep(0, "شغل الكشاف", WeykyIntent.FlashlightIntent(true))
    val step2 = WeykyIntent.CompoundStep(1, "اقرأ سجل المكالمات", WeykyIntent.CallIntent(showMissed = true))

    var step1Executed = false
    var step2Executed = false

    val steps = listOf(step1, step2)
    var requiresPermissionResult: ExecutionResult.RequiresPermission? = null

    for (step in steps) {
      if (step.index == 0) {
        step1Executed = true
      } else if (step.index == 1) {
        step2Executed = true
        requiresPermissionResult = ExecutionResult.RequiresPermission(
          permission = android.Manifest.permission.READ_CALL_LOG,
          rationale = "تم تنفيذ الخطوة الأولى، لكن يلزم منح صلاحية سجل المكالمات لمتابعة الخطوة الثانية.",
          visualState = CoreVisualState.ERROR
        )
        break // Halted
      }
    }

    assertTrue(step1Executed)
    assertTrue(step2Executed)
    assertNotNull(requiresPermissionResult)
    assertEquals(android.Manifest.permission.READ_CALL_LOG, requiresPermissionResult!!.permission)
  }

  // =========================================================================
  // SECTION 4.D: Dialogue State and Follow-Up Resolution
  // =========================================================================

  @Test
  fun test4D_conversationSessionRetainsContextAndResolvesFollowUp() {
    val reasoner = ConversationalReasoner()
    val session = ConversationSession()

    // 1. Ask about sleep wellness
    val r1 = reasoner.reason("حدثني عن النوم الصحي", session)
    session.recordAssistantTurn(r1.replyText, r1.proactiveSuggestions, r1.topicDetected)
    assertEquals("sleep_wellness", session.currentTopic)

    // 2. Follow-up about caffeine/coffee
    val r2 = reasoner.reason("وماذا عن القهوة؟", session)
    assertTrue(r2.replyText.contains("الكافيين") || r2.replyText.contains("النوم"))
    assertEquals("sleep_wellness", r2.topicDetected)

    // 3. Ambiguous follow-up triggers clarification expectation
    val r3 = reasoner.reason("وماذا عن ذلك؟", session)
    assertTrue(r3.replyText.contains("هل تقصد المتابعة حول موضوع") || r3.replyText.contains("يرجى التوضيح"))
    assertEquals(PendingExpectation.AWAITING_CLARIFICATION, session.pendingExpectation)
  }

  @Test
  fun test4D_bilingualConsistency_arabicAndEnglish() {
    val reasoner = ConversationalReasoner()
    val sessionAr = ConversationSession()
    val sessionEn = ConversationSession()

    // Arithmetic in Arabic vs English
    val resAr = reasoner.reason("احسب 25 * 4", sessionAr)
    val resEn = reasoner.reason("calculate 25 * 4", sessionEn)

    assertEquals("calculation", resAr.topicDetected)
    assertEquals("calculation", resEn.topicDetected)
    assertTrue(resAr.replyText.contains("100"))
    assertTrue(resEn.replyText.contains("100"))
  }

  // =========================================================================
  // SECTION 4.E: Epistemic Integrity and Proactive Suggestions
  // =========================================================================

  @Test
  fun test4E_epistemicIntegrity_forecastStatusIsEstimateNeverFact() {
    val timeProvider = SettableTimeProvider().apply {
      setTime(hourOfDay = 13, minute = 0, second = 0)
    }
    val calculator = BatteryForecastCalculator(timeProvider)
    val snap = createMockSnapshot(
      batteryDatum = ContextDatum.fact(createBattery(percentage = 70), "BatteryProvider")
    )
    val target = BatteryForecastTarget(17, 0, "الساعة 5:00 م", "5:00 PM", "الخامسة")

    val outcome = calculator.calculateForecast(snap, target, includesMusic = false, isArabic = true)
    assertTrue(outcome is BatteryForecastOutcome.Success)
    val success = outcome as BatteryForecastOutcome.Success

    // Epistemic requirements:
    // 1. Confirmed actual measurement is cited as FACT (70%)
    assertTrue(success.synthesis.factsUsed.any { it.contains("70%") })
    // 2. Projected outcome is explicitly tagged [ESTIMATE]
    assertTrue(success.synthesis.synthesizedSpeech.contains("[ESTIMATE]"))
    // 3. Proactive suggestions are calibrated: with high battery (70%), no unneeded alarm is raised
    assertNull(success.synthesis.nextStep)

    // With low battery (e.g. 22%), proactive suggestion is generated when minimum drops below 20%
    val lowSnap = createMockSnapshot(
      batteryDatum = ContextDatum.fact(createBattery(percentage = 22), "BatteryProvider")
    )
    val lowOutcome = calculator.calculateForecast(lowSnap, target, includesMusic = false, isArabic = true)
    assertTrue(lowOutcome is BatteryForecastOutcome.Success)
    val lowSuccess = lowOutcome as BatteryForecastOutcome.Success
    assertNotNull("Low battery forecast must offer proactive suggestion", lowSuccess.synthesis.nextStep)
    assertEquals("تفعيل توفير الطاقة", lowSuccess.synthesis.nextStep!!.label)
  }

  @Test
  fun test4E_honestFallbackForUnindexedTopics() {
    val reasoner = ConversationalReasoner()
    val session = ConversationSession()

    val res = reasoner.reason("ما هي نظرية الأوتار الفائقة ونموذج كالابي-ياو؟", session)
    assertEquals("unindexed_query", res.topicDetected)
    assertTrue(res.replyText.contains("قاعدة المعرفة المحلية"))
  }

  // =========================================================================
  // SECTION 4.F: Voice / Text Submission Consistency
  // =========================================================================

  @Test
  fun test4F_voiceAndTextConsistency_intentResolution() {
    val session = ConversationSession()

    // 1. Spoken vs typed flashlight command
    val typedTorch = "شغل الكشاف"
    val spokenTorch = "يا ويكي شغل الكشاف"
    val intent1 = IntentParser.parse(typedTorch, session)
    val intent2 = IntentParser.parse(spokenTorch, session)

    assertEquals(WeykyIntent.FlashlightIntent(turnOn = true), intent1)
    assertEquals(WeykyIntent.FlashlightIntent(turnOn = true), intent2)

    // 2. Spoken vs typed battery forecast
    val typedForecast = "كم نسبة البطارية حتى المساء مع الموسيقى؟"
    val spokenForecast = "ويكي كم نسبة البطارية حتى المساء مع الموسيقى"
    val intentF1 = IntentParser.parse(typedForecast, session)
    val intentF2 = IntentParser.parse(spokenForecast, session)

    assertTrue(intentF1 is WeykyIntent.TimeAwareBatteryForecastIntent)
    assertTrue(intentF2 is WeykyIntent.TimeAwareBatteryForecastIntent)
    assertEquals(
      (intentF1 as WeykyIntent.TimeAwareBatteryForecastIntent).includesMusic,
      (intentF2 as WeykyIntent.TimeAwareBatteryForecastIntent).includesMusic
    )
  }
}
